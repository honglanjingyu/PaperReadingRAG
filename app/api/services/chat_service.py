# app/api/services/chat_service.py
"""
聊天服务 - 支持会话记忆
"""

from typing import List, Dict, Any, Optional
import json
import asyncio
import random
import os
from concurrent.futures import ThreadPoolExecutor

from app.service.core.rag import (
    enhanced_search_with_hybrid_and_rerank,
    generate_answer,
    generate_answer_stream
)

# ========== 新增：导入评估模块 ==========
from app.service.core.evaluation import get_evaluator

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

        # ========== 评估模块配置 ==========
        self._evaluator = None
        self._eval_enabled = os.getenv("ENABLE_EVAL", "false").lower() == "true"
        self._eval_sample_rate = float(os.getenv("EVAL_SAMPLE_RATE", "0.1"))  # 10%采样率
        self._eval_timeout = int(os.getenv("EVAL_TIMEOUT", "3"))  # 3秒超时

        if self._eval_enabled and MEMORY_AVAILABLE:
            self._evaluator = get_evaluator() if MEMORY_AVAILABLE else None
            if self._evaluator and self._evaluator.is_eval_available():
                print(f"✅ 评估模块已启用（采样率: {self._eval_sample_rate * 100}%）")
            elif self._evaluator:
                print("⚠️ 评估模块已初始化，但 LLM 评估不可用（将使用简单评估）")
        else:
            print("ℹ️ 评估模块已禁用")

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

    async def _async_evaluate(
            self,
            question: str,
            answer: str,
            results: List[Dict],
            session_id: str,
            rewritten_query: str = None
    ):
        """
        后台异步执行评估（不阻塞主响应）

        Args:
            question: 原始问题
            answer: 生成的答案
            results: 检索结果
            session_id: 会话ID
            rewritten_query: 改写后的问题
        """
        try:
            # 提取上下文（用于评估）
            contexts = []
            for r in results[:5]:  # 最多取5个上下文
                content = r.get("content", "") or r.get("content_with_weight", "")
                if content:
                    contexts.append(content[:1000])  # 限制长度

            if not contexts:
                print(f"⚠️ 评估跳过 [{session_id[:8]}]: 无上下文")
                return

            # 使用超时控制执行评估
            loop = asyncio.get_event_loop()

            # 创建带超时的任务
            try:
                eval_result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda: self._evaluator.evaluate_full(
                            question=question,
                            answer=answer,
                            retrieved_docs=results,
                            contexts=contexts,
                            session_id=session_id
                        )
                    ),
                    timeout=self._eval_timeout
                )

                # 可选：在控制台打印简要评估结果（调试用）
                if eval_result.get("generation_metrics"):
                    gm = eval_result["generation_metrics"]
                    print(f"\n📊 评估 [{session_id[:8]}]: "
                          f"Faith={gm.get('faithfulness', 0):.2f}, "
                          f"Rel={gm.get('answer_relevancy', 0):.2f}, "
                          f"耗时={eval_result.get('eval_time', 0):.2f}s")

            except asyncio.TimeoutError:
                print(f"⚠️ 评估超时 [{session_id[:8]}]: 超过 {self._eval_timeout} 秒")

        except Exception as e:
            print(f"⚠️ 后台评估失败 [{session_id[:8]}]: {e}")

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
            user_level: str = None
    ) -> Dict[str, Any]:
        """问答处理 - 支持会话记忆"""

        actual_session_id = None
        if enable_memory and self._memory_manager:
            actual_session_id = self._get_or_create_session(session_id)
        else:
            actual_session_id = session_id or "default"

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
            verbose=False,
            user_level=user_level
        )

        if not retrieval_result.get("success"):
            return {
                "success": False,
                "question": question,
                "session_id": actual_session_id,
                "error": retrieval_result.get("error", "检索失败")
            }

        results = retrieval_result.get("results", [])

        # 没有检索结果时的处理
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

        # ========== 异步评估（不阻塞响应） ==========
        if self._eval_enabled and self._evaluator and answer and results:
            # 采样率控制：只对部分请求进行评估
            if random.random() < self._eval_sample_rate:
                # 创建后台任务，不等待结果
                asyncio.create_task(
                    self._async_evaluate(
                        question=question,
                        answer=answer,
                        results=results,
                        session_id=actual_session_id,
                        rewritten_query=rewritten_query
                    )
                )
            else:
                print(f"ℹ️ 跳过评估 [{actual_session_id[:8]}]: 未命中采样")

        # 更新会话记忆
        if enable_memory and self._memory_manager and actual_session_id and actual_session_id != "default" and answer:
            self._memory_manager.add_message(actual_session_id, "user", question)
            self._memory_manager.add_message(actual_session_id, "assistant", answer)

        # ========== 返回结果（不包含评估指标） ==========
        return {
            "success": True,
            "question": question,
            "session_id": actual_session_id,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "has_history": has_history,
            "results": results[:top_k],  # 只返回 top_k 个结果
            "retrieval_info": {
                "total_recalled": retrieval_result.get("total_recalled", 0),
                "total_returned": retrieval_result.get("total_returned", 0),
                "enable_rerank": enable_rerank,
                "enable_query_rewrite": enable_query_rewrite,
                "rerank_model": retrieval_result.get("rerank_model", "unknown")
            },
            "model_info": generation_result.get("model_info", {})
            # ❌ 注意：这里没有添加 evaluation 字段，避免响应卡顿
        }

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
        actual_session_id = None
        if enable_memory and self._memory_manager:
            actual_session_id = self._get_or_create_session(session_id)
        else:
            actual_session_id = session_id or "default"

        if not results:
            yield "未找到相关文档"
            return

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

        # 收集完整答案
        answer_chunks = []
        for chunk in sync_generate():
            answer_chunks.append(chunk)
            yield chunk

        full_answer = "".join(answer_chunks)

        # ========== 异步评估（流式完成后） ==========
        if self._eval_enabled and self._evaluator and full_answer and results:
            if random.random() < self._eval_sample_rate:
                asyncio.create_task(
                    self._async_evaluate(
                        question=question,
                        answer=full_answer,
                        results=results,
                        session_id=actual_session_id
                    )
                )

        # 更新记忆
        if enable_memory and self._memory_manager and actual_session_id and actual_session_id != "default" and full_answer:
            self._memory_manager.add_message(actual_session_id, "user",
                                             question.split('改写:')[-1].strip() if '改写:' in question else question)
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