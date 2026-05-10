# app/api/services/chat_service.py
"""
聊天服务 - 支持会话记忆
"""

from typing import List, Dict, Any, Optional
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.service.core.rag import (
    enhanced_search_with_hybrid_and_rerank,
    generate_answer,
    generate_answer_stream
)

try:
    from app.service.core.memory import RedisSessionMemory, MemoryInjector, get_memory_manager
    MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"❌ 记忆模块导入失败: {e}，请确保 Redis 服务已启动")
    MEMORY_AVAILABLE = False
    RedisSessionMemory = None


class ChatService:
    """聊天服务 - 支持会话记忆"""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)
        # 初始化记忆组件
        self._memory_manager = None
        self._memory_injector = None
        if MEMORY_AVAILABLE:
            self._init_memory()

    def _init_memory(self):
        """初始化记忆组件"""
        try:
            self._memory_manager = get_memory_manager()
            self._memory_injector = MemoryInjector(self._memory_manager)
            print("✅ 短期记忆已启用")
        except Exception as e:
            print(f"⚠️ 短期记忆初始化失败: {e}")

    def _get_or_create_session(self, session_id: str = None) -> str:
        """获取或创建会话"""
        if self._memory_manager:
            return self._memory_manager.get_or_create_session(session_id)
        return session_id or "default"

    async def ask(
            self,
            question: str,
            session_id: str = None,
            history: Optional[List[Dict[str, str]]] = None,
            top_k: int = 5,
            recall_k: int = 10,
            similarity_threshold: float = 0.3,
            enable_rerank: bool = True,
            enable_query_rewrite: bool = True,
            template_name: str = "detailed",
            keyword_weight: float = 0.4,
            vector_weight: float = 0.6,
            rerank_type: str = "remote",
            index_name: str = "rag_documents",
            enable_memory: bool = True,
            user_level: str = None  # 新增参数
    ) -> Dict[str, Any]:
        """问答处理 - 支持会话记忆和用户等级"""

        actual_session_id = None
        if enable_memory and self._memory_manager:
            actual_session_id = self._get_or_create_session(session_id)
        else:
            actual_session_id = session_id or "default"

        # 执行增强检索（传递用户等级）
        retrieval_result = enhanced_search_with_hybrid_and_rerank(
            question=question,
            index_name=index_name,
            recall_k=recall_k,
            top_k=top_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            enable_rerank=enable_rerank,
            enable_query_rewrite=enable_query_rewrite,
            similarity_threshold=similarity_threshold,
            rerank_type=rerank_type,
            verbose=False,
            user_level=user_level  # 传递用户等级
        )

        if not retrieval_result.get("success"):
            return {
                "success": False,
                "question": question,
                "session_id": actual_session_id,
                "error": retrieval_result.get("error", "检索失败")
            }

        results = retrieval_result.get("results", [])
        if not results:
            return {
                "success": False,
                "question": question,
                "session_id": actual_session_id,
                "answer": "未找到与问题相关的文档内容，请尝试其他问题或上传更多相关文档。",
                "results": [],
                "retrieval_info": retrieval_result
            }

        rewritten_query = retrieval_result.get("rewritten_query", question)

        # 获取历史信息
        has_history = False
        if enable_memory and self._memory_injector and actual_session_id and actual_session_id != "default":
            history_text = self._memory_manager.get_history_text(actual_session_id, max_turns=10)
            has_history = bool(history_text)

        # 生成答案
        generation_result = generate_answer(
            question=rewritten_query,
            results=results,
            history=history,
            template_name=template_name,
            verbose=False
        )

        if not generation_result.get("success"):
            return {
                "success": False,
                "question": question,
                "session_id": actual_session_id,
                "error": generation_result.get("error", "生成失败")
            }

        answer = generation_result.get("answer", "")

        # 更新会话记忆
        if enable_memory and self._memory_manager and actual_session_id and actual_session_id != "default" and answer:
            self._memory_manager.add_message(actual_session_id, "user", question)
            self._memory_manager.add_message(actual_session_id, "assistant", answer)

        return {
            "success": True,
            "question": question,
            "session_id": actual_session_id,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "has_history": has_history,
            "results": results,
            "retrieval_info": {
                "total_recalled": retrieval_result.get("total_recalled", 0),
                "total_returned": retrieval_result.get("total_returned", 0),
                "enable_rerank": enable_rerank,
                "enable_query_rewrite": enable_query_rewrite,
                "rerank_model": retrieval_result.get("rerank_model", "unknown")
            },
            "model_info": generation_result.get("model_info", {})
        }

    async def ask_stream(
            self,
            question: str,
            session_id: str = None,
            history: Optional[List[Dict[str, str]]] = None,
            top_k: int = 5,
            recall_k: int = 10,
            template_name: str = "detailed",
            index_name: str = "rag_documents",
            enable_memory: bool = True,
            user_level: str = None
    ):
        """流式问答处理 - 支持会话记忆"""

        # 获取或创建会话
        actual_session_id = None
        if enable_memory and self._memory_manager:
            actual_session_id = self._get_or_create_session(session_id)
        else:
            actual_session_id = session_id or "default"

        # 执行检索
        loop = asyncio.get_event_loop()

        retrieval_result = await loop.run_in_executor(
            self._executor,
            lambda: enhanced_search_with_hybrid_and_rerank(
                question=question,
                index_name=index_name,
                recall_k=recall_k,
                top_k=top_k,
                verbose=False,
                user_level=user_level
            )
        )

        if not retrieval_result.get("success") or not retrieval_result.get("results"):
            yield "未找到相关文档"
            return

        results = retrieval_result.get("results", [])
        rewritten_query = retrieval_result.get("rewritten_query", question)

        # 流式生成答案
        full_answer = ""

        def sync_generate():
            nonlocal full_answer
            for chunk in generate_answer_stream(
                    question=rewritten_query,
                    results=results,
                    history=history,
                    template_name=template_name,
                    verbose=False
            ):
                if chunk and chunk.strip():
                    full_answer += chunk
                    yield chunk

        for chunk in sync_generate():
            yield chunk

        # 更新记忆
        if enable_memory and self._memory_manager and actual_session_id and actual_session_id != "default" and full_answer:
            self._memory_manager.add_message(actual_session_id, "user", question)
            self._memory_manager.add_message(actual_session_id, "assistant", full_answer)

    async def ask_stream_with_results(
            self,
            question: str,
            session_id: str = None,
            history: Optional[List[Dict[str, str]]] = None,
            results: List[Dict[str, Any]] = None,
            template_name: str = "detailed",
            enable_memory: bool = True
    ):
        """流式问答处理 - 使用已有的检索结果"""

        # 获取或创建会话
        actual_session_id = None
        if enable_memory and self._memory_manager:
            actual_session_id = self._get_or_create_session(session_id)
        else:
            actual_session_id = session_id or "default"

        if not results:
            yield "未找到相关文档"
            return

        # 流式生成答案
        full_answer = ""

        def sync_generate():
            nonlocal full_answer
            for chunk in generate_answer_stream(
                    question=question,
                    results=results,
                    history=history,
                    template_name=template_name,
                    verbose=False
            ):
                if chunk and chunk.strip():
                    full_answer += chunk
                    yield chunk

        for chunk in sync_generate():
            yield chunk

        # 更新记忆
        if enable_memory and self._memory_manager and actual_session_id and actual_session_id != "default" and full_answer:
            self._memory_manager.add_message(actual_session_id, "user", question.split('改写:')[-1].strip() if '改写:' in question else question)
            self._memory_manager.add_message(actual_session_id, "assistant", full_answer)

    async def search_only(
            self,
            question: str,
            top_k: int = 5,
            recall_k: int = 10,
            index_name: str = "rag_documents"
    ) -> Dict[str, Any]:
        """仅检索"""
        result = enhanced_search_with_hybrid_and_rerank(
            question=question,
            index_name=index_name,
            recall_k=recall_k,
            top_k=top_k,
            verbose=False
        )

        return {
            "success": result.get("success", False),
            "question": question,
            "rewritten_query": result.get("rewritten_query"),
            "total_recalled": result.get("total_recalled", 0),
            "total_returned": result.get("total_returned", 0),
            "results": result.get("results", []),
            "error": result.get("error")
        }

    async def generate_only(
            self,
            question: str,
            results: List[Dict[str, Any]],
            history: Optional[List[Dict[str, str]]] = None,
            template_name: str = "detailed"
    ) -> Dict[str, Any]:
        """仅生成"""
        result = generate_answer(
            question=question,
            results=results,
            history=history,
            template_name=template_name,
            verbose=False
        )

        return {
            "success": result.get("success", False),
            "question": question,
            "answer": result.get("answer"),
            "error": result.get("error")
        }

    def clear_session(self, session_id: str) -> bool:
        """清除会话记忆"""
        if self._memory_manager:
            return self._memory_manager.clear_session(session_id)
        return False

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        if self._memory_manager:
            return self._memory_manager.get_session_info(session_id)
        return None


__all__ = ['ChatService']