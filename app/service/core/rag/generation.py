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
        verbose: bool = True,
        preview_answer: bool = False  # 新增：控制是否打印答案预览
) -> dict:
    """
    生成模块：将构造好的 Prompt 发送给 LLM，生成最终答案

    Args:
        question: 用户问题（建议使用 Query 改写后的问题）
        results: 检索结果列表
        history: 对话历史
        template_name: 模板名称
        model_type: 模型类型
        verbose: 是否打印详细信息
        preview_answer: 是否在函数内打印答案（默认 True）
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

    if verbose and answer and preview_answer:
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


def test_llm_generation(questions: List[str] = None, stream: bool = False, use_enhanced_search: bool = True):
    """
    测试 LLM 生成功能

    Args:
        questions: 问题列表
        stream: 是否流式输出
        use_enhanced_search: 是否使用增强检索（包含 Query 改写），默认 True
    """
    from .utils import TEST_QUESTIONS
    from .search import search_similar_documents, enhanced_search_with_hybrid_and_rerank

    if questions is None:
        questions = TEST_QUESTIONS

    import os

    # 读取配置
    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    recall_top_k = int(os.getenv("SIMILARITY_TOP_K", "10"))
    rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    # 确保 recall_top_k >= rerank_top_k
    if recall_top_k < rerank_top_k:
        print(f"\n⚠ 警告: SIMILARITY_TOP_K({recall_top_k}) < RERANK_TOP_K({rerank_top_k})")
        print(f"  已自动调整: SIMILARITY_TOP_K = RERANK_TOP_K = {rerank_top_k}")
        recall_top_k = rerank_top_k

    for question in questions:
        print(f"\n{'=' * 70}")
        print(f"原始问题: {question}")
        print("=" * 70)

        # 根据配置选择检索方式
        if use_enhanced_search:
            # 使用增强检索（包含 Query 改写）
            print("\n使用增强检索模式...")

            enhanced_result = enhanced_search_with_hybrid_and_rerank(
                question=question,
                index_name=index_name,
                recall_k=recall_top_k,
                top_k=rerank_top_k,
                keyword_weight=float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4")),
                vector_weight=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")),
                enable_rerank=os.getenv("ENABLE_RERANK", "true").lower() == "true",
                enable_query_rewrite=True,
                similarity_threshold=similarity_threshold,
                rerank_type=os.getenv("RERANK_TYPE", "remote"),
                verbose=False
            )

            if not enhanced_result.get("success") or not enhanced_result.get("results"):
                print("未找到相关文档，跳过生成")
                continue

            # 使用改写后的问题
            rewritten_query = enhanced_result.get("rewritten_query", question)
            results = enhanced_result.get("results", [])

            print(f"\n改写后问题: {rewritten_query}")
            print(f"检索结果数: {len(results)}")

            question_for_llm = rewritten_query

        else:
            # 使用传统检索（不使用 Query 改写）
            print("\n使用传统检索模式...")

            search_result = search_similar_documents(
                question=question,
                es_index_name=index_name,
                top_k=rerank_top_k,
                similarity_threshold=similarity_threshold,
                verbose=False
            )

            if not search_result.get("success") or not search_result.get("results"):
                print("未找到相关文档，跳过生成")
                continue

            results = search_result.get("results", [])
            question_for_llm = question

        # 执行生成
        if stream:
            for _ in generate_answer_stream(
                    question=question_for_llm,
                    results=results,
                    template_name="detailed",
                    verbose=False
            ):
                pass
        else:
            # verbose=True 会打印答案，preview_answer=False 避免重复
            result = generate_answer(
                question=question_for_llm,
                results=results,
                template_name="detailed",
                verbose=True,
                preview_answer=True  # 在函数内打印完整答案
            )

            if result.get("success"):
                print(f"\n✓ 生成成功")
                print(f"  模型: {result['model_info']['model_name']}")
                # 不再打印答案预览，因为已经在函数内打印过了
            else:
                print(f"\n✗ 生成失败")


def test_llm_generation_with_rewritten_only(questions: List[str] = None, stream: bool = False):
    """
    专门使用 Query 改写后的问题进行测试（不使用原始问题）

    Args:
        questions: 问题列表
        stream: 是否流式输出
    """
    from .utils import TEST_QUESTIONS
    from .search import enhanced_search_with_hybrid_and_rerank

    if questions is None:
        questions = TEST_QUESTIONS

    import os

    # 读取配置
    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    recall_top_k = int(os.getenv("SIMILARITY_TOP_K", "10"))
    rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    print("\n" + "=" * 70)
    print("专门使用 Query 改写后的问题进行生成测试")
    print("=" * 70)

    for question in questions:
        print(f"\n{'=' * 70}")
        print(f"原始问题: {question}")
        print("=" * 70)

        # 使用增强检索获取改写后的问题
        enhanced_result = enhanced_search_with_hybrid_and_rerank(
            question=question,
            index_name=index_name,
            recall_k=recall_top_k,
            top_k=rerank_top_k,
            keyword_weight=float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4")),
            vector_weight=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")),
            enable_rerank=os.getenv("ENABLE_RERANK", "true").lower() == "true",
            enable_query_rewrite=True,
            similarity_threshold=similarity_threshold,
            rerank_type=os.getenv("RERANK_TYPE", "remote"),
            verbose=False
        )

        if not enhanced_result.get("success") or not enhanced_result.get("results"):
            print("未找到相关文档，跳过生成")
            continue

        # 只使用改写后的问题
        rewritten_query = enhanced_result.get("rewritten_query", question)
        results = enhanced_result.get("results", [])

        print(f"\n改写后问题（将直接传给 LLM）: {rewritten_query}")
        print(f"检索结果数: {len(results)}")

        # 执行生成
        if stream:
            for _ in generate_answer_stream(
                    question=rewritten_query,
                    results=results,
                    template_name="detailed",
                    verbose=False
            ):
                pass
        else:
            # verbose=True 会打印答案，preview_answer=True 打印完整答案
            result = generate_answer(
                question=rewritten_query,
                results=results,
                template_name="detailed",
                verbose=True,
                preview_answer=True  # 在函数内打印，避免外部重复打印
            )

            if result.get("success"):
                print(f"\n✓ 生成成功")
                print(f"  模型: {result['model_info']['model_name']}")
                # ❌ 删除这里的答案预览，避免重复
                # ✅ 答案已经在 generate_answer 函数内打印过了
            else:
                print(f"\n✗ 生成失败")