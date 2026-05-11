# app/service/core/evaluation/__init__.py
"""
评估模块 - RAG 系统评估
"""

from .evaluator import RAGEvaluator, get_evaluator
from .metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    compute_hit_at_k,
    compute_mrr,
    compute_faithfulness,
    compute_answer_relevancy,
    compute_context_recall,
    compute_context_precision
)
from .logger import EvaluationLogger

__all__ = [
    'RAGEvaluator',
    'get_evaluator',
    'RetrievalMetrics',
    'GenerationMetrics',
    'compute_hit_at_k',
    'compute_mrr',
    'compute_faithfulness',
    'compute_answer_relevancy',
    'compute_context_recall',
    'compute_context_precision',
    'EvaluationLogger'
]