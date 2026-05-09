# app/service/core/rag/generation.py

"""
RAG 生成模块
包含：上下文构造 -> 推理生成
"""

from typing import List, Dict, Optional
from dotenv import load_dotenv
import os

load_dotenv()

from app.service.core.prompt import PromptBuilder
from app.service.core.llm import get_llm_service


def generate_answer(
        question: str,
        results: List[Dict],
        history: List[Dict] = None,
        template_name: str = "detailed",
        model_type: str = None,
        verbose: bool = False,
        preview_answer: bool = False
) -> dict:
    """
    生成模块：将构造好的 Prompt 发送给 LLM，生成最终答案（非流式）
    """
    prompt_builder = PromptBuilder(max_context_length=4000, include_scores=True)
    messages = prompt_builder.build_messages(
        question=question,
        results=results,
        history=history,
        template_name=template_name
    )

    llm_service = get_llm_service(model_type)
    answer = llm_service.generate(messages, verbose=verbose)

    if verbose and answer and preview_answer:
        print("\n" + "-" * 50)
        print("LLM 生成结果:")
        print("-" * 50)
        print(answer)
        print("-" * 50)

    return {
        "success": answer is not None,
        "question": question,
        "messages": messages,
        "answer": answer,
        "model_info": llm_service.get_model_info()
    }


def generate_answer_stream(
        question: str,
        results: List[Dict],
        history: List[Dict] = None,
        template_name: str = "detailed",
        model_type: str = None,
        verbose: bool = False
):
    """流式生成答案"""
    prompt_builder = PromptBuilder(max_context_length=4000, include_scores=True)
    messages = prompt_builder.build_messages(
        question=question,
        results=results,
        history=history,
        template_name=template_name
    )

    llm_service = get_llm_service(model_type)

    for chunk in llm_service.generate_stream(messages, verbose=verbose):
        if chunk:
            yield chunk


__all__ = [
    'generate_answer',
    'generate_answer_stream'
]