# app/api/routes/chat.py
"""
智能问答路由 - 支持会话记忆
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List
import json

from app.api.models import ChatRequest, GenerateRequest
from app.api.config import settings
from app.api.dependencies import get_chat_service

router = APIRouter()


@router.post("/chat/ask")
async def ask_question(request: ChatRequest) -> Dict[str, Any]:
    """
    问答接口 - 完整流程，支持会话记忆

    流程：用户问题向量化 -> 相似度搜索 -> 增强搜索 -> 上下文构造 -> 推理生成

    使用session_id可以保持对话记忆：
    - 首次请求不传session_id，系统自动生成并返回
    - 后续请求传入返回的session_id即可保持上下文

    Args:
        request: 聊天请求参数
            - question: 用户问题
            - session_id: 会话ID（可选，用于保持对话记忆）
            - history: 对话历史（可选，传统方式）
            - top_k: 返回结果数量
            - recall_k: 召回数量
            - similarity_threshold: 相似度阈值
            - enable_rerank: 是否启用重排序
            - enable_query_rewrite: 是否启用查询改写
            - template_name: Prompt模板名称
            - enable_memory: 是否启用短期记忆

    Returns:
        包含答案和会话信息的响应
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
            session_id=request.session_id,
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
            index_name=settings.index_name,
            enable_memory=request.enable_memory
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")


@router.post("/chat/ask/stream")
async def ask_question_stream(request: ChatRequest):
    """
    流式问答接口 - 支持会话记忆

    实时流式返回答案，适合聊天界面使用

    Args:
        request: 聊天请求参数（同上）

    Returns:
        Server-Sent Events (SSE) 流式响应
    """
    top_k = request.top_k or settings.rerank_top_k
    recall_k = request.recall_k or settings.similarity_top_k

    if recall_k < top_k:
        recall_k = top_k

    async def generate():
        try:
            chat_service = get_chat_service()

            # 先发送开始标记
            yield json.dumps({"type": "start", "content": "", "session_id": request.session_id}) + "\n"

            # 收集完整答案用于更新记忆
            full_answer = ""
            session_id = request.session_id

            # 流式生成答案
            async for chunk in chat_service.ask_stream(
                    question=request.question,
                    session_id=session_id,
                    history=request.history,
                    top_k=top_k,
                    recall_k=recall_k,
                    template_name=request.template_name,
                    index_name=settings.index_name,
                    enable_memory=request.enable_memory
            ):
                if chunk:
                    full_answer += chunk
                    yield json.dumps({"type": "answer", "content": chunk}) + "\n"

            # 发送结束标记，包含session_id（如果是新创建的）
            yield json.dumps({
                "type": "end",
                "content": "",
                "session_id": session_id,
                "full_answer_length": len(full_answer)
            }) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/chat/search")
async def search_only(request: ChatRequest) -> Dict[str, Any]:
    """
    仅检索接口 - 只返回相关文档，不生成答案

    适合需要查看检索结果的场景
    """
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
    """
    仅生成接口 - 基于已有检索结果生成答案

    适合已经手动选定了检索结果的场景
    """
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


@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str) -> Dict[str, Any]:
    """
    清除会话记忆

    删除指定会话的所有对话历史，释放内存

    Args:
        session_id: 要清除的会话ID

    Returns:
        操作结果
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id不能为空")

    try:
        chat_service = get_chat_service()
        success = chat_service.clear_session(session_id)

        if success:
            return {
                "success": True,
                "session_id": session_id,
                "message": f"会话 {session_id} 已清除"
            }
        else:
            return {
                "success": False,
                "session_id": session_id,
                "message": f"会话 {session_id} 不存在或清除失败"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除会话失败: {str(e)}")


@router.get("/chat/session/{session_id}")
async def get_session_info(session_id: str) -> Dict[str, Any]:
    """
    获取会话信息

    查看指定会话的状态，包括对话轮次、消息数量等

    Args:
        session_id: 会话ID

    Returns:
        会话详细信息
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id不能为空")

    try:
        chat_service = get_chat_service()
        info = chat_service.get_session_info(session_id)

        if info:
            return {
                "success": True,
                "session_id": session_id,
                "info": info
            }
        else:
            return {
                "success": False,
                "session_id": session_id,
                "message": f"会话 {session_id} 不存在"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话信息失败: {str(e)}")


@router.get("/chat/sessions")
async def list_active_sessions() -> Dict[str, Any]:
    """
    列出所有活跃会话

    获取当前系统中所有活跃会话的统计信息
    """
    try:
        chat_service = get_chat_service()

        # 获取记忆管理器
        from app.service.core.memory import get_memory_manager
        memory_manager = get_memory_manager()

        return {
            "success": True,
            "active_sessions_count": memory_manager.get_active_sessions_count(),
            "total_sessions_count": memory_manager.get_total_sessions_count(),
            "session_ttl": memory_manager.session_ttl,
            "max_turns": memory_manager.max_turns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


# 可选：支持多轮对话的快捷接口
@router.post("/chat/conversation")
async def conversation(
        question: str,
        session_id: Optional[str] = None,
        enable_memory: bool = True
) -> Dict[str, Any]:
    """
    简化的对话接口

    只需传入问题和可选的session_id，其他参数使用默认值

    Args:
        question: 用户问题
        session_id: 会话ID（可选）
        enable_memory: 是否启用记忆

    Returns:
        答案和会话信息
    """
    try:
        chat_service = get_chat_service()

        # 使用默认配置
        result = await chat_service.ask(
            question=question,
            session_id=session_id,
            top_k=settings.rerank_top_k,
            recall_k=settings.similarity_top_k,
            similarity_threshold=settings.similarity_threshold,
            enable_rerank=settings.enable_rerank,
            enable_query_rewrite=settings.enable_query_rewrite,
            template_name="detailed",
            keyword_weight=settings.keyword_weight,
            vector_weight=settings.vector_weight,
            rerank_type=settings.rerank_type,
            index_name=settings.index_name,
            enable_memory=enable_memory
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")


__all__ = ['router']