# app/api/services/chat_service.py (更新版 - 添加记忆支持)
"""
聊天服务 - 添加短期记忆支持
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
from app.service.core.memory import SessionMemory, MemoryInjector, get_memory_manager


class ChatService:
    """聊天服务 - 支持会话记忆"""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)
        # 初始化记忆管理器
        self._memory_manager: Optional[SessionMemory] = None
        self._memory_injector: Optional[MemoryInjector] = None
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
            session_id: str = None,  # 新增：会话ID
            history: Optional[List[Dict[str, str]]] = None,  # 保留兼容性
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
            enable_memory: bool = True  # 新增：是否启用记忆
    ) -> Dict[str, Any]:
        """
        问答处理 - 支持会话记忆

        Args:
            question: 用户问题
            session_id: 会话ID（用于记忆）
            history: 对话历史（传统方式，优先使用session_id）
            ...
            enable_memory: 是否启用短期记忆
        """
        # 获取或创建会话
        if enable_memory and self._memory_manager:
            session_id = self._get_or_create_session(session_id)
        else:
            session_id = None

        # 执行增强检索
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
            verbose=False
        )

        if not retrieval_result.get("success"):
            return {
                "success": False,
                "question": question,
                "session_id": session_id,
                "error": retrieval_result.get("error", "检索失败")
            }

        results = retrieval_result.get("results", [])
        if not results:
            return {
                "success": False,
                "question": question,
                "session_id": session_id,
                "answer": "未找到与问题相关的文档内容，请尝试其他问题或上传更多相关文档。",
                "results": [],
                "retrieval_info": retrieval_result
            }

        rewritten_query = retrieval_result.get("rewritten_query", question)

        # 获取历史上下文
        context_info = {"has_history": False, "history_text": ""}
        if enable_memory and self._memory_injector and session_id:
            # 使用记忆注入器构建包含历史的Prompt
            context_info = self._memory_injector.inject_into_prompt(
                session_id=session_id,
                question=rewritten_query,
                context="",  # 会在生成时处理
                template_name=template_name,
                max_history_turns=10
            )
            # 如果有历史，将历史信息添加到检索结果中供生成使用
            if context_info.get("has_history"):
                history_text = context_info.get("history_text", "")
                # 创建一个特殊的系统消息来提供历史上下文
                history_context = {
                    "content": f"## 对话历史\n{history_text}\n\n## 当前文档内容",
                    "is_history": True
                }

        # 生成答案（传入历史）
        generation_result = generate_answer(
            question=rewritten_query,
            results=results,
            history=history,  # 如果有传入的history，使用它
            template_name=template_name,
            verbose=False
        )

        if not generation_result.get("success"):
            return {
                "success": False,
                "question": question,
                "session_id": session_id,
                "error": generation_result.get("error", "生成失败")
            }

        answer = generation_result.get("answer", "")

        # 更新会话记忆
        if enable_memory and self._memory_injector and session_id and answer:
            self._memory_injector.update_memory(session_id, question, answer)

        return {
            "success": True,
            "question": question,
            "session_id": session_id,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "has_history": context_info.get("has_history", False) if enable_memory else False,
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
            session_id: str = None,  # 新增：会话ID
            history: Optional[List[Dict[str, str]]] = None,
            top_k: int = 5,
            recall_k: int = 10,
            template_name: str = "detailed",
            index_name: str = "rag_documents",
            enable_memory: bool = True  # 新增：是否启用记忆
    ):
        """流式问答处理 - 支持会话记忆"""

        # 获取或创建会话
        if enable_memory and self._memory_manager:
            session_id = self._get_or_create_session(session_id)

        # 执行检索
        loop = asyncio.get_event_loop()

        retrieval_result = await loop.run_in_executor(
            self._executor,
            lambda: enhanced_search_with_hybrid_and_rerank(
                question=question,
                index_name=index_name,
                recall_k=recall_k,
                top_k=top_k,
                verbose=False
            )
        )

        if not retrieval_result.get("success") or not retrieval_result.get("results"):
            yield "未找到相关文档"
            return

        results = retrieval_result.get("results", [])
        rewritten_query = retrieval_result.get("rewritten_query", question)

        # 获取历史上下文
        history_text = ""
        if enable_memory and self._memory_injector and session_id:
            history_text = self._memory_injector.format_history(session_id, max_turns=10)

        # 流式生成答案（在线程池中执行同步生成器）
        def sync_generate():
            # 构建带历史的prompt
            full_answer = ""
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

            # 更新记忆
            if enable_memory and self._memory_injector and session_id and full_answer:
                self._memory_injector.update_memory(session_id, question, full_answer)

        for chunk in sync_generate():
            yield chunk

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