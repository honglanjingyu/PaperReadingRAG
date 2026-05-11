# app/service/core/evaluation/evaluator.py
"""
RAG 评估器 - 整合所有评估指标
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from .metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    compute_hit_at_k,
    compute_mrr,
    compute_faithfulness,
    compute_answer_relevancy,
    compute_context_recall,
    compute_context_precision,
    get_eval_llm
)
from .logger import EvaluationLogger

logger = logging.getLogger(__name__)

# 配置开关
ENABLE_EVAL_METRICS_IN_RESPONSE = os.getenv("ENABLE_EVAL_METRICS_IN_RESPONSE", "true").lower() == "true"


class RAGEvaluator:
    """
    RAG 评估器
    支持检索层评估和生成层评估
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logger = EvaluationLogger()
        self.eval_llm = get_eval_llm()
        logger.info(f"RAG 评估器初始化完成, LLM可用: {self.eval_llm.is_available()}")

    def evaluate_retrieval(
        self,
        question: str,
        retrieved_docs: List[Dict],
        relevant_doc_ids: Set[str],
        k: int = 5
    ) -> RetrievalMetrics:
        """
        评估检索层性能

        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档列表
            relevant_doc_ids: 相关文档的ID集合
            k: Hit@K 中的 K 值

        Returns:
            RetrievalMetrics 对象
        """
        metrics = RetrievalMetrics(k=k)

        if retrieved_docs and relevant_doc_ids:
            metrics.hit_at_k = compute_hit_at_k(retrieved_docs, relevant_doc_ids, k)
            metrics.mrr = compute_mrr(retrieved_docs, relevant_doc_ids)

        logger.debug(f"检索评估完成: Hit@{k}={metrics.hit_at_k:.4f}, MRR={metrics.mrr:.4f}")
        return metrics

    def evaluate_generation(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> GenerationMetrics:
        """
        评估生成层质量

        Args:
            question: 用户问题
            answer: 生成的答案
            contexts: 上下文文档列表（检索结果的文本内容）

        Returns:
            GenerationMetrics 对象
        """
        metrics = GenerationMetrics()

        if not answer:
            logger.warning("答案为 None，无法评估生成层")
            return metrics

        if not contexts:
            logger.warning("上下文为空，生成评估结果可能不准确")

        # 计算各项指标
        metrics.faithfulness = compute_faithfulness(question, answer, contexts)
        metrics.answer_relevancy = compute_answer_relevancy(question, answer)
        metrics.context_recall = compute_context_recall(question, answer, contexts)
        metrics.context_precision = compute_context_precision(question, contexts)

        logger.debug(
            f"生成评估完成: Faithfulness={metrics.faithfulness:.4f}, "
            f"AnswerRelevancy={metrics.answer_relevancy:.4f}, "
            f"ContextRecall={metrics.context_recall:.4f}, "
            f"ContextPrecision={metrics.context_precision:.4f}"
        )

        return metrics

    def evaluate_full(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Dict],
        relevant_doc_ids: Optional[Set[str]] = None,
        contexts: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        完整评估（检索 + 生成）

        Args:
            question: 用户问题
            answer: 生成的答案
            retrieved_docs: 检索到的文档列表
            relevant_doc_ids: 相关文档ID集合（可选）
            contexts: 上下文文本列表（可选，默认从 retrieved_docs 提取）
            session_id: 会话ID
            k: Hit@K 中的 K 值

        Returns:
            包含所有评估指标的字典
        """
        result = {
            "question": question,
            "answer": answer[:500] if answer else None,  # 截断避免日志过长
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "retrieval_metrics": {},
            "generation_metrics": {},
            "full_answer": answer  # 完整答案
        }

        # 提取上下文
        if contexts is None and retrieved_docs:
            contexts = []
            for doc in retrieved_docs[:5]:
                content = doc.get("content") or doc.get("content_with_weight") or ""
                if content:
                    contexts.append(content)

        # 评估检索层（如果有相关文档信息）
        if relevant_doc_ids:
            retrieval_metrics = self.evaluate_retrieval(question, retrieved_docs, relevant_doc_ids, k)
            result["retrieval_metrics"] = retrieval_metrics.to_dict()

        # 评估生成层
        if answer:
            generation_metrics = self.evaluate_generation(question, answer, contexts or [])
            result["generation_metrics"] = generation_metrics.to_dict()

        # 记录到日志文件
        self.logger.log_evaluation(
            question=question,
            answer=answer,
            retrieval_metrics=result["retrieval_metrics"],
            generation_metrics=result["generation_metrics"],
            session_id=session_id
        )

        # 可选：将评估指标添加到响应中
        if ENABLE_EVAL_METRICS_IN_RESPONSE:
            result["include_in_response"] = True

        return result

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        search_func,
        generate_func,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        批量评估测试用例

        Args:
            test_cases: 测试用例列表，每个包含 question, expected_answer, relevant_doc_ids
            search_func: 检索函数，接收 question 返回检索结果
            generate_func: 生成函数，接收 question, retrieved_docs 返回答案
            k: Hit@K 中的 K 值

        Returns:
            批量评估结果汇总
        """
        results = []
        total_retrieval = RetrievalMetrics()
        total_generation = GenerationMetrics()
        count = 0

        for test_case in test_cases:
            question = test_case.get("question")
            relevant_doc_ids = set(test_case.get("relevant_doc_ids", []))

            if not question:
                continue

            # 执行检索
            retrieved_docs = search_func(question) if search_func else []

            # 生成答案
            answer = generate_func(question, retrieved_docs) if generate_func else None

            # 评估
            eval_result = self.evaluate_full(
                question=question,
                answer=answer,
                retrieved_docs=retrieved_docs,
                relevant_doc_ids=relevant_doc_ids,
                k=k
            )

            results.append(eval_result)
            count += 1

            # 累加指标
            if eval_result.get("retrieval_metrics"):
                total_retrieval.hit_at_k += eval_result["retrieval_metrics"].get("hit_at_k", 0)
                total_retrieval.mrr += eval_result["retrieval_metrics"].get("mrr", 0)
            if eval_result.get("generation_metrics"):
                total_generation.faithfulness += eval_result["generation_metrics"].get("faithfulness", 0)
                total_generation.answer_relevancy += eval_result["generation_metrics"].get("answer_relevancy", 0)
                total_generation.context_recall += eval_result["generation_metrics"].get("context_recall", 0)
                total_generation.context_precision += eval_result["generation_metrics"].get("context_precision", 0)

        # 计算平均值
        if count > 0:
            avg_retrieval = RetrievalMetrics(
                hit_at_k=total_retrieval.hit_at_k / count,
                mrr=total_retrieval.mrr / count
            )
            avg_generation = GenerationMetrics(
                faithfulness=total_generation.faithfulness / count,
                answer_relevancy=total_generation.answer_relevancy / count,
                context_recall=total_generation.context_recall / count,
                context_precision=total_generation.context_precision / count
            )
        else:
            avg_retrieval = RetrievalMetrics()
            avg_generation = GenerationMetrics()

        # 记录批量评估汇总
        self.logger.log_batch_summary(
            total_cases=count,
            avg_retrieval_metrics=avg_retrieval.to_dict(),
            avg_generation_metrics=avg_generation.to_dict()
        )

        return {
            "total_cases": count,
            "average_retrieval_metrics": avg_retrieval.to_dict(),
            "average_generation_metrics": avg_generation.to_dict(),
            "results": results
        }

    def is_eval_available(self) -> bool:
        """检查评估是否可用"""
        return self.eval_llm.is_available()


# 全局单例
_evaluator = None


def get_evaluator() -> RAGEvaluator:
    """获取评估器实例"""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator


__all__ = ['RAGEvaluator', 'get_evaluator']