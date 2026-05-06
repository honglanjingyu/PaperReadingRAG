# app/api/services/chat_service.py
"""
聊天服务
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


class ChatService:
    """聊天服务"""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def ask(
            self,
            question: str,
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
            index_name: str = "rag_documents"
    ) -> Dict[str, Any]:
        """问答处理"""
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
                "error": retrieval_result.get("error", "检索失败")
            }

        results = retrieval_result.get("results", [])
        if not results:
            return {
                "success": False,
                "question": question,
                "answer": "未找到与问题相关的文档内容，请尝试其他问题或上传更多相关文档。",
                "results": [],
                "retrieval_info": retrieval_result
            }

        # 使用改写后的问题进行生成
        rewritten_query = retrieval_result.get("rewritten_query", question)

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
                "error": generation_result.get("error", "生成失败")
            }

        return {
            "success": True,
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": generation_result.get("answer", ""),
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
            history: Optional[List[Dict[str, str]]] = None,
            top_k: int = 5,
            recall_k: int = 10,
            template_name: str = "detailed",
            index_name: str = "rag_documents"
    ):
        """流式问答处理"""
        # 执行检索（在线程池中执行，避免阻塞事件循环）
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

        # 流式生成答案 - 在线程池中执行同步生成器
        def sync_generate():
            for chunk in generate_answer_stream(
                    question=rewritten_query,
                    results=results,
                    history=history,
                    template_name=template_name,
                    verbose=False
            ):
                if chunk and chunk.strip():
                    yield chunk

        # 将同步生成器转换为异步生成器
        for chunk in sync_generate():
            yield chunk

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