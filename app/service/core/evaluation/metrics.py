# app/service/core/evaluation/metrics.py
"""
评估指标计算模块
包含检索层指标和生成层指标
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# ========== 评估 LLM 配置 ==========
from dotenv import load_dotenv
load_dotenv()

ENABLE_LLM_EVAL = os.getenv("ENABLE_LLM_EVAL", "true").lower() == "true"
EVALUATION_TYPE = os.getenv("EVALUATION_TYPE", "remote")
EVALUATION_MODEL = os.getenv("EVALUATION_MODEL", "qwen-plus-2025-07-28")
EVALUATION_BASE_URL = os.getenv("EVALUATION_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EVALUATION_API_KEY = os.getenv("EVALUATION_API_KEY") or os.getenv("MODEL_API_KEY")
LOCAL_EVALUATION_PATH = os.getenv("LOCAL_EVALUATION_PATH", "")


@dataclass
class RetrievalMetrics:
    """检索层评估指标"""
    hit_at_k: float = 0.0
    mrr: float = 0.0
    recall_at_k: float = 0.0
    k: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hit_at_k": round(self.hit_at_k, 4),
            "mrr": round(self.mrr, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "k": self.k
        }


@dataclass
class GenerationMetrics:
    """生成层评估指标"""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_recall": round(self.context_recall, 4),
            "context_precision": round(self.context_precision, 4)
        }


def compute_hit_at_k(
    retrieved_docs: List[Dict],
    relevant_doc_ids: Set[str],
    k: int = 5
) -> float:
    """
    计算 Hit@K
    判断前K个检索结果中是否包含相关文档

    Args:
        retrieved_docs: 检索到的文档列表
        relevant_doc_ids: 相关文档的ID集合
        k: 考虑的前K个结果

    Returns:
        Hit@K 分数 (0 或 1，或平均值)
    """
    if not retrieved_docs or not relevant_doc_ids:
        return 0.0

    top_k_docs = retrieved_docs[:k]
    for doc in top_k_docs:
        doc_id = doc.get("chunk_id") or doc.get("_id") or doc.get("id")
        if doc_id and doc_id in relevant_doc_ids:
            return 1.0

    return 0.0


def compute_mrr(
    retrieved_docs: List[Dict],
    relevant_doc_ids: Set[str]
) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)
    第一个相关文档的倒数排名平均值

    Args:
        retrieved_docs: 检索到的文档列表
        relevant_doc_ids: 相关文档的ID集合

    Returns:
        MRR 分数
    """
    if not retrieved_docs or not relevant_doc_ids:
        return 0.0

    for rank, doc in enumerate(retrieved_docs, 1):
        doc_id = doc.get("chunk_id") or doc.get("_id") or doc.get("id")
        if doc_id and doc_id in relevant_doc_ids:
            return 1.0 / rank

    return 0.0


class EvaluationLLM:
    """评估专用 LLM 客户端"""

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
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化 LLM 客户端"""
        if not ENABLE_LLM_EVAL:
            logger.info("LLM 评估已禁用")
            return

        if EVALUATION_TYPE == "remote":
            self._init_remote()
        elif EVALUATION_TYPE == "local":
            self._init_local()
        else:
            self._init_remote()

    def _init_remote(self):
        """初始化远程 LLM"""
        try:
            from openai import OpenAI

            if not EVALUATION_API_KEY:
                logger.warning("评估 API Key 未配置")
                return

            self._client = OpenAI(
                api_key=EVALUATION_API_KEY,
                base_url=EVALUATION_BASE_URL
            )
            logger.info(f"评估 LLM 初始化成功: {EVALUATION_MODEL}")
        except Exception as e:
            logger.error(f"评估 LLM 初始化失败: {e}")

    def _init_local(self):
        """初始化本地 LLM"""
        try:
            from sentence_transformers import SentenceTransformer

            if LOCAL_EVALUATION_PATH:
                self._local_model = SentenceTransformer(LOCAL_EVALUATION_PATH)
            else:
                self._local_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("本地评估模型初始化成功")
        except Exception as e:
            logger.error(f"本地评估模型初始化失败: {e}")
            self._local_model = None

    def is_available(self) -> bool:
        """检查评估 LLM 是否可用"""
        return ENABLE_LLM_EVAL and (self._client is not None or self._local_model is not None)

    def generate(self, prompt: str) -> Optional[str]:
        """生成评估结果"""
        if not self.is_available():
            return None

        if EVALUATION_TYPE == "remote" and self._client:
            return self._generate_remote(prompt)
        elif EVALUATION_TYPE == "local" and hasattr(self, '_local_model'):
            # 本地模型暂不支持直接生成，返回 None
            return None
        return None

    def _generate_remote(self, prompt: str) -> Optional[str]:
        """远程生成"""
        try:
            response = self._client.chat.completions.create(
                model=EVALUATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"评估 LLM 调用失败: {e}")
            return None

    def get_score_from_response(self, response: str) -> float:
        """从响应中提取分数（0-1）"""
        if not response:
            return 0.0

        # 尝试提取数值
        patterns = [
            r'分数[：:]\s*([0-9.]+)',
            r'score[：:]\s*([0-9.]+)',
            r'([0-9.]+)\s*分',
            r'([0-9.]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = float(match.group(1))
                # 确保分数在 0-1 范围内
                if score > 1:
                    score = score / 10 if score <= 10 else 1.0
                return max(0.0, min(1.0, score))

        # 基于关键词判断
        positive_keywords = ['准确', '正确', '相关', 'faithful', 'relevant', '是']
        negative_keywords = ['不准确', '错误', '不相关', 'not faithful', 'irrelevant', '否']

        response_lower = response.lower()
        positive_score = sum(1 for kw in positive_keywords if kw in response_lower)
        negative_score = sum(1 for kw in negative_keywords if kw in response_lower)

        if positive_score + negative_score > 0:
            return positive_score / (positive_score + negative_score)

        return 0.5  # 默认中等分数


# 全局评估 LLM 实例
_eval_llm = None


def get_eval_llm():
    """获取评估 LLM 实例"""
    global _eval_llm
    if _eval_llm is None:
        _eval_llm = EvaluationLLM()
    return _eval_llm


# app/service/core/evaluation/metrics.py

def compute_faithfulness(
        question: str,
        answer: str,
        contexts: List[str]
) -> float:
    """
    计算 Faithfulness（忠实度）- 更严格的评估
    """
    if not answer or not contexts:
        return 0.0

    eval_llm = get_eval_llm()

    # 如果没有 LLM 评估，使用基于上下文的匹配
    if not eval_llm.is_available():
        return _compute_faithfulness_strict(answer, contexts)

    # 使用 LLM 评估（更严格）
    context_str = "\n---\n".join(contexts[:3])
    if len(context_str) > 2000:
        context_str = context_str[:2000] + "..."

    prompt = f"""请严格评估答案是否忠实于上下文。只输出 0-1 之间的数字。

## 上下文
{context_str}

## 问题
{question}

## 答案
{answer}

## 评估标准
- 1.0: 答案中的每个关键信息都能在上下文中找到
- 0.7: 大部分信息有依据，但有少量推测
- 0.4: 部分信息无法在上下文中验证
- 0.0: 答案中有明显的事实错误或编造

注意：如果答案中出现了上下文中没有的信息，必须扣分。

分数（只输出数字）："""

    response = eval_llm.generate(prompt)
    return eval_llm.get_score_from_response(response)


def _compute_faithfulness_strict(answer: str, contexts: List[str]) -> float:
    """基于关键词匹配的严格忠实度评估"""
    if not answer or not contexts:
        return 0.0

    import jieba

    # 提取答案中的关键句子（按句号分割）
    sentences = answer.replace('。', '。\n').split('\n')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if not sentences:
        return 0.5

    all_context = " ".join(contexts).lower()

    # 统计有多少句子可以在上下文中找到支持
    supported_sentences = 0
    for sentence in sentences:
        # 提取句子中的关键词
        words = jieba.lcut_for_search(sentence)
        # 过滤短词和停用词
        keywords = [w for w in words if len(w) > 1 and w not in ['什么', '如何', '为什么', '请问']]

        if not keywords:
            supported_sentences += 0.5
            continue

        # 检查关键词在上下文中出现的比例
        matched = sum(1 for kw in keywords if kw in all_context)
        ratio = matched / len(keywords)

        if ratio >= 0.7:
            supported_sentences += 1
        elif ratio >= 0.4:
            supported_sentences += 0.5

    score = supported_sentences / len(sentences) if sentences else 0.5
    return min(0.95, max(0.05, score))


def compute_answer_relevancy(
        question: str,
        answer: str
) -> float:
    """
    计算 Answer Relevancy（答案相关性）- 更严格的评估
    """
    if not answer:
        return 0.0

    # 如果答案太短，可能相关性不高
    if len(answer) < 20:
        return 0.3

    eval_llm = get_eval_llm()
    if not eval_llm.is_available():
        return _compute_answer_relevancy_strict(question, answer)

    prompt = f"""请严格评估答案是否直接回答了问题。

## 问题
{question}

## 答案
{answer}

## 评估标准
- 1.0: 答案直接、完整地回答了问题
- 0.7: 答案回答了问题，但不够完整
- 0.4: 答案与问题部分相关
- 0.0: 答案与问题完全不相关

注：如果答案只是说"根据文档内容..."但没有具体信息，应该给低分。

分数（只输出数字）："""

    response = eval_llm.generate(prompt)
    return eval_llm.get_score_from_response(response)


def _compute_answer_relevancy_strict(question: str, answer: str) -> float:
    """基于关键词匹配的严格相关性评估"""
    if not answer:
        return 0.0

    import jieba

    # 提取问题关键词
    q_words = set(jieba.lcut_for_search(question))
    q_words = {w for w in q_words if len(w) > 1}

    # 提取答案关键词
    a_words = set(jieba.lcut_for_search(answer))
    a_words = {w for w in a_words if len(w) > 1}

    if not q_words:
        return 0.5

    # 计算交集比例
    overlap = len(q_words & a_words)
    score = min(1.0, overlap / len(q_words))

    # 降级因子：答案太短降权
    if len(answer) < 50:
        score *= 0.7

    # 降级因子：答案中没有具体内容
    if "根据" in answer and len(answer) < 100:
        score *= 0.8

    return max(0.1, score)


def _compute_answer_relevancy_simple(question: str, answer: str) -> float:
    """基于关键词匹配的简单答案相关性评估"""
    if not answer:
        return 0.0

    import jieba

    question_words = set(jieba.lcut_for_search(question))
    answer_words = set(jieba.lcut_for_search(answer))

    question_words = {w for w in question_words if len(w) > 1}
    answer_words = {w for w in answer_words if len(w) > 1}

    if not question_words:
        return 0.5

    intersection = question_words & answer_words
    if not intersection:
        return 0.1

    return min(1.0, len(intersection) / len(question_words))


def compute_context_recall(
    question: str,
    answer: str,
    contexts: List[str]
) -> float:
    """
    计算 Context Recall（上下文召回率）
    评估答案需要的信息是否被上下文完整覆盖

    Args:
        question: 用户问题
        answer: 生成的答案
        contexts: 上下文文档列表

    Returns:
        上下文召回率分数 (0-1)
    """
    if not answer or not contexts:
        return 0.0

    eval_llm = get_eval_llm()
    if not eval_llm.is_available():
        return _compute_context_recall_simple(answer, contexts)

    context_str = "\n---\n".join(contexts[:3])
    if len(context_str) > 3000:
        context_str = context_str[:3000] + "..."

    prompt = f"""请评估回答问题所需的信息是否都被上下文覆盖。

## 上下文文档
{context_str}

## 问题
{question}

## 答案
{answer}

## 评估标准
评估答案中的信息要点有多少可以在上下文中找到：
- 所需信息完全被覆盖
- 所需信息部分被覆盖
- 所需信息很少被覆盖

请只给出分数（0-1之间，精确到小数点后2位）：
分数："""

    response = eval_llm.generate(prompt)
    return eval_llm.get_score_from_response(response)


def _compute_context_recall_simple(answer: str, contexts: List[str]) -> float:
    """基于关键词匹配的简单上下文召回率评估"""
    if not answer or not contexts:
        return 0.0

    import jieba

    answer_words = set(jieba.lcut_for_search(answer))
    answer_words = {w for w in answer_words if len(w) > 1}

    if not answer_words:
        return 0.5

    all_context = " ".join(contexts).lower()
    matched_count = sum(1 for w in answer_words if w.lower() in all_context)

    return min(1.0, matched_count / len(answer_words))


def compute_context_precision(
    question: str,
    contexts: List[str]
) -> float:
    """
    计算 Context Precision（上下文精确率）
    评估检索到的上下文中相关内容的比例

    Args:
        question: 用户问题
        contexts: 上下文文档列表

    Returns:
        上下文精确率分数 (0-1)
    """
    if not contexts:
        return 0.0

    eval_llm = get_eval_llm()
    if not eval_llm.is_available():
        return _compute_context_precision_simple(question, contexts)

    # 取前5个上下文进行评估
    contexts_to_eval = contexts[:5]
    context_str = "\n---\n".join(contexts_to_eval)
    if len(context_str) > 3000:
        context_str = context_str[:3000] + "..."

    prompt = f"""请评估以下上下文文档与问题的相关程度。

## 问题
{question}

## 上下文文档
{context_str}

## 评估标准
逐一评估每个文档是否与问题相关：
- 高度相关：直接包含答案
- 部分相关：包含部分相关信息
- 不相关：与问题无关

请只给出分数（0-1之间，精确到小数点后2位）：
分数："""

    response = eval_llm.generate(prompt)
    return eval_llm.get_score_from_response(response)


def _compute_context_precision_simple(question: str, contexts: List[str]) -> float:
    """基于关键词匹配的简单上下文精确率评估"""
    if not contexts:
        return 0.0

    import jieba

    question_words = set(jieba.lcut_for_search(question))
    question_words = {w for w in question_words if len(w) > 1}

    if not question_words:
        return 0.5

    scores = []
    for context in contexts:
        context_lower = context.lower()
        matched = sum(1 for w in question_words if w.lower() in context_lower)
        score = min(1.0, matched / len(question_words))
        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


__all__ = [
    'RetrievalMetrics',
    'GenerationMetrics',
    'compute_hit_at_k',
    'compute_mrr',
    'compute_faithfulness',
    'compute_answer_relevancy',
    'compute_context_recall',
    'compute_context_precision'
]