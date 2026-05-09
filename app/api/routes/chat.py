# app/api/routes/chat.py
"""
智能问答路由 - 支持会话记忆和 URL 参数传递 session_id
"""

from fastapi import APIRouter, HTTPException, Query, Header
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List
import json
import asyncio
import logging

from app.api.models import ChatRequest, GenerateRequest
from app.api.config import settings
from app.api.dependencies import get_chat_service
from app.service.core.rag import enhanced_search_with_hybrid_and_rerank
from app.auth.jwt_utils import get_user_id_from_token
from app.db.database import get_db_manager

logger = logging.getLogger(__name__)
router = APIRouter()


# app/api/routes/chat.py
@router.get("/chat/session/create")
async def create_new_session(
        authorization: Optional[str] = Header(None),
        user_id: str = "default"
) -> Dict[str, Any]:
    """
    创建新会话，返回 session_id
    """
    try:
        # 获取登录用户ID
        token_user_id = None
        if authorization:
            token = authorization[7:] if authorization.startswith("Bearer ") else authorization
            token_user_id = get_user_id_from_token(token)

        chat_service = get_chat_service()
        if hasattr(chat_service, '_memory_manager') and chat_service._memory_manager:
            # 使用登录用户ID或默认值
            effective_user_id = str(token_user_id) if token_user_id else user_id
            session_id = chat_service._memory_manager.get_or_create_session(user_id=effective_user_id)
        else:
            from app.service.core.memory import get_memory_manager
            memory_manager = get_memory_manager()
            effective_user_id = str(token_user_id) if token_user_id else user_id
            session_id = memory_manager.get_or_create_session(user_id=effective_user_id)

        # 如果用户已登录，关联会话到数据库
        if token_user_id:
            db = get_db_manager()
            db.associate_session(token_user_id, session_id)

        return {
            "success": True,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")

# 修改 get_session_info 接口
@router.get("/chat/session/{session_id}")
async def get_session_info(
        session_id: str,
        authorization: Optional[str] = Header(None),
        user_id: str = "default"
) -> Dict[str, Any]:
    """获取会话信息 - 验证权限"""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id不能为空")

    # 验证权限（仅当提供 token 时）
    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        token_user_id = get_user_id_from_token(token)
        if token_user_id:
            db = get_db_manager()
            if not db.verify_session_access(token_user_id, session_id):
                raise HTTPException(status_code=403, detail="无权访问此会话")

    try:
        from app.service.core.memory import get_memory_manager
        memory = get_memory_manager()

        info = memory.get_session_info(session_id, user_id)

        if info:
            return {
                "success": True,
                "session_id": session_id,
                "info": {
                    "session_id": info.get("session_id"),
                    "turn_count": info.get("turn_count", 0),
                    "message_count": info.get("message_count", 0),
                    "created_at": info.get("created_at"),
                    "last_accessed": info.get("last_accessed"),
                    "is_active": info.get("is_active", True)
                }
            }
        else:
            return {
                "success": False,
                "session_id": session_id,
                "info": None,
                "message": f"会话 {session_id} 不存在"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话信息失败: {str(e)}")

@router.get("/chat/session/{session_id}/history")
async def get_session_history(
        session_id: str,
        limit: int = Query(50, ge=1, le=200),
        authorization: Optional[str] = Header(None),
        user_id: str = "default"
) -> Dict[str, Any]:
    """获取会话的完整历史记录"""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id不能为空")

    # 验证权限（仅当提供 token 时）
    if authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        token_user_id = get_user_id_from_token(token)
        if token_user_id:
            db = get_db_manager()
            if not db.verify_session_access(token_user_id, session_id):
                raise HTTPException(status_code=403, detail="无权访问此会话")

    try:
        from app.service.core.memory import get_memory_manager
        memory = get_memory_manager()
        history = memory.get_session_history(session_id, user_id, limit=limit)

        return {
            "success": True,
            "session_id": session_id,
            "message_count": len(history),
            "messages": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")


@router.get("/chat/session/{session_id}/messages")
async def get_session_messages(
        session_id: str,
        max_turns: int = Query(20, ge=1, le=50),
        user_id: str = "default"
) -> Dict[str, Any]:
    """
    获取会话消息（用于恢复对话界面）

    Args:
        session_id: 会话ID
        max_turns: 最大轮次数
        user_id: 用户ID

    Returns:
        会话消息列表
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id不能为空")

    try:
        from app.service.core.memory import get_memory_manager
        memory = get_memory_manager()

        # 检查会话是否存在
        info = memory.get_session_info(session_id, user_id)
        if not info:
            return {
                "success": False,
                "error": "会话不存在"
            }

        messages = memory.get_conversation_history(session_id, user_id, max_turns=max_turns)

        return {
            "success": True,
            "session_id": session_id,
            "session_info": info,
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息失败: {str(e)}")


# app/api/routes/chat.py
@router.post("/chat/ask")
async def ask_question(
        request: ChatRequest,
        authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    问答接口 - 完整流程，支持会话记忆
    """
    # 验证会话权限（如果提供了 session_id）
    if request.session_id and authorization:
        token = authorization[7:] if authorization.startswith("Bearer ") else authorization
        token_user_id = get_user_id_from_token(token)
        if token_user_id:
            db = get_db_manager()
            if not db.verify_session_access(token_user_id, request.session_id):
                raise HTTPException(status_code=403, detail="无权访问此会话")
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
    """
    top_k = request.top_k or settings.rerank_top_k
    recall_k = request.recall_k or settings.similarity_top_k
    similarity_threshold = request.similarity_threshold or settings.similarity_threshold
    enable_rerank = request.enable_rerank if request.enable_rerank is not None else settings.enable_rerank
    enable_query_rewrite = request.enable_query_rewrite if request.enable_query_rewrite is not None else settings.enable_query_rewrite

    if recall_k < top_k:
        recall_k = top_k

    async def generate():
        try:
            chat_service = get_chat_service()

            # 发送开始标记
            yield json.dumps({"type": "start", "content": "", "session_id": request.session_id}) + "\n"

            full_answer = ""
            session_id = request.session_id

            # 1. 先执行检索
            loop = asyncio.get_event_loop()

            retrieval_result = await loop.run_in_executor(
                chat_service._executor,
                lambda: enhanced_search_with_hybrid_and_rerank(
                    question=request.question,
                    index_name=settings.index_name,
                    recall_k=recall_k,
                    top_k=top_k,
                    keyword_weight=settings.keyword_weight,
                    vector_weight=settings.vector_weight,
                    enable_rerank=enable_rerank,
                    enable_query_rewrite=enable_query_rewrite,
                    similarity_threshold=similarity_threshold,
                    rerank_type=settings.rerank_type,
                    verbose=False
                )
            )

            # 2. 发送检索结果到前端
            if retrieval_result.get("success") and retrieval_result.get("results"):
                results = retrieval_result.get("results", [])
                formatted_results = []
                for r in results[:top_k]:
                    formatted_results.append({
                        "content": r.get("content", ""),
                        "score": r.get("score", 0),
                        "document_name": r.get("document_name", "")
                    })

                yield json.dumps({
                    "type": "retrieval_results",
                    "results": formatted_results,
                    "retrieval_info": {
                        "total_recalled": retrieval_result.get("total_recalled", 0),
                        "total_returned": len(formatted_results),
                        "enable_rerank": enable_rerank,
                        "enable_query_rewrite": enable_query_rewrite
                    }
                }) + "\n"

                # 如果没有检索结果，直接结束
                if not results:
                    yield json.dumps({
                        "type": "end",
                        "content": "",
                        "session_id": session_id,
                        "no_results": True
                    }) + "\n"
                    return

                # 3. 流式生成答案
                rewritten_query = retrieval_result.get("rewritten_query", request.question)

                async for chunk in chat_service.ask_stream_with_results(
                        question=rewritten_query,
                        session_id=session_id,
                        history=request.history,
                        results=results,
                        template_name=request.template_name,
                        enable_memory=request.enable_memory
                ):
                    if chunk:
                        full_answer += chunk
                        yield json.dumps({"type": "answer", "content": chunk}) + "\n"
            else:
                # 检索失败
                yield json.dumps({
                    "type": "error",
                    "content": retrieval_result.get("error", "检索失败")
                }) + "\n"

            # 发送结束标记
            yield json.dumps({
                "type": "end",
                "content": "",
                "session_id": session_id,
                "full_answer_length": len(full_answer)
            }) + "\n"

        except Exception as e:
            logger.error(f"流式问答失败: {e}")
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


@router.get("/chat/sessions")
async def list_active_sessions() -> Dict[str, Any]:
    """
    列出所有活跃会话

    获取当前系统中所有活跃会话的统计信息
    """
    try:
        from app.service.core.memory import get_memory_manager
        memory_manager = get_memory_manager()

        return {
            "success": True,
            "active_sessions_count": memory_manager.get_active_sessions_count(),
            "total_sessions_count": memory_manager.get_total_sessions_count(),
            "session_ttl": getattr(memory_manager, 'session_ttl', 3600),
            "max_turns": getattr(memory_manager, 'max_turns', 20)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@router.post("/chat/conversation")
async def conversation(
        question: str,
        session_id: Optional[str] = None,
        enable_memory: bool = True
) -> Dict[str, Any]:
    """
    简化的对话接口

    只需传入问题和可选的session_id，其他参数使用默认值
    """
    try:
        chat_service = get_chat_service()

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