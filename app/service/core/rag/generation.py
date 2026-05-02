# app/service/core/rag/generation.py
"""
RAG 生成模块
包含：上下文构造 -> 推理生成
"""

from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from app.service.core.prompt import PromptBuilder
from app.service.core.llm import get_llm_service


def generate_answer(
    question: str,
    results: List[Dict],
    history: List[Dict] = None,
    template_name: str = "detailed",
    model_type: str = None,
    verbose: bool = True
) -> dict:
    """
    生成模块：将构造好的 Prompt 发送给 LLM，生成最终答案
    """
    if verbose:
        print("\n[12/12] 生成模块 - 构造提示词并调用大语言模型...")
        print("=" * 70)

    prompt_builder = PromptBuilder(max_context_length=4000, include_scores=True)
    messages = prompt_builder.build_messages(
        question=question,
        results=results,
        history=history,
        template_name=template_name
    )

    if verbose:
        print(f"\n构造的 Prompt 预览:")
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            print(f"  [{role}]: {content}...")

    llm_service = get_llm_service(model_type)
    answer = llm_service.generate(messages, verbose=verbose)

    if verbose and answer:
        print("\n" + "-" * 70)
        print("LLM 生成结果:")
        print("-" * 70)
        print(answer)
        print("-" * 70)

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
    verbose: bool = True
):
    """流式生成答案"""
    if verbose:
        print("\n[12/12] 生成模块 - 流式调用大语言模型...")
        print("=" * 70)

    prompt_builder = PromptBuilder(max_context_length=4000, include_scores=True)
    messages = prompt_builder.build_messages(
        question=question,
        results=results,
        history=history,
        template_name=template_name
    )

    llm_service = get_llm_service(model_type)
    print("\n回答: ", end="", flush=True)

    for chunk in llm_service.generate_stream(messages, verbose=verbose):
        print(chunk, end="", flush=True)
        yield chunk
    print("\n")


def test_llm_generation(questions: List[str] = None, stream: bool = False):
    """测试 LLM 生成功能"""
    from .utils import TEST_QUESTIONS
    from .search import search_similar_documents

    if questions is None:
        questions = TEST_QUESTIONS

    import os
    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    top_k = int(os.getenv("SIMILARITY_TOP_K", "5"))

    for question in questions:
        print(f"\n{'=' * 70}")
        print(f"问题: {question}")
        print("=" * 70)

        search_result = search_similar_documents(
            question=question,
            es_index_name=index_name,
            top_k=top_k,
            similarity_threshold=0.3,
            verbose=False
        )

        if not search_result.get("success") or not search_result.get("results"):
            print("未找到相关文档，跳过生成")
            continue

        if stream:
            for _ in generate_answer_stream(
                question=question,
                results=search_result["results"],
                template_name="detailed",
                verbose=False
            ):
                pass
        else:
            result = generate_answer(
                question=question,
                results=search_result["results"],
                template_name="detailed",
                verbose=True
            )

            if result.get("success"):
                print(f"\n✓ 生成成功")
                print(f"  模型: {result['model_info']['model_name']}")
            else:
                print(f"\n✗ 生成失败")