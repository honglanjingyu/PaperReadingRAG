# app/service/core/rag/utils.py
"""
RAG 工具函数和测试
"""

import os
from typing import List, Dict

# 全局测试问题列表
TEST_QUESTIONS = [
    "世运电子的主要业务是什么？",
]


# 向后兼容的分块器类
class RecursiveChunker:
    """向后兼容的分块器类"""

    def __init__(self, chunk_token_num: int = 256):
        from app.service.core.chunking import RecursiveChunkerSimple
        self.chunker = RecursiveChunkerSimple(chunk_token_num=chunk_token_num)

    def chunk(self, text: str) -> List[str]:
        return self.chunker.chunk(text)


def run_all_tests():
    """运行所有测试 - 优化版本，避免重复操作"""
    from .search import (
        test_user_question_vectorization,
        test_enhanced_retrieval,
    )
    from .generation import test_llm_generation_with_rewritten_only

    print("\n" + "=" * 70)
    print("运行所有测试")
    print("=" * 70)

    # 从环境变量读取配置
    recall_top_k = int(os.getenv("SIMILARITY_TOP_K", "10"))
    rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))

    # 健壮性检查
    if rerank_top_k > recall_top_k:
        print(f"\n⚠ 警告: RERANK_TOP_K({rerank_top_k}) > SIMILARITY_TOP_K({recall_top_k})")
        print(f"  已自动调整: SIMILARITY_TOP_K = RERANK_TOP_K = {rerank_top_k}")
        recall_top_k = rerank_top_k

    if recall_top_k <= 0:
        recall_top_k = 5
    if rerank_top_k <= 0:
        rerank_top_k = recall_top_k

    # 1. 用户问题向量化测试（轻量级，保留）
    print("\n" + "=" * 70)
    print("1. 用户问题向量化测试")
    print("=" * 70)
    test_user_question_vectorization()

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    # 检查索引是否存在
    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(index_name)

        if not index_exists or search_service.es_store.get_document_count(index_name) == 0:
            print(f"\n⚠ 索引 '{index_name}' 不存在或为空，跳过检索和生成测试")
            return

    except Exception as e:
        print(f"\n检查索引时出错: {e}")
        return

    # 2. 增强检索测试（包含搜索和重排序）
    print("\n" + "=" * 70)
    print(f"2. 增强检索测试 (召回{recall_top_k}个 -> Rerank -> 返回{rerank_top_k}个)")
    print("=" * 70)

    # 只执行一次增强检索，保存结果供生成测试使用
    enhanced_results = []
    for question in TEST_QUESTIONS[:1]:
        print(f"\n测试问题: {question}")
        result = test_enhanced_retrieval_single(
            question=question,
            index_name=index_name,
            recall_k=recall_top_k,
            top_k=rerank_top_k,
            similarity_threshold=similarity_threshold,
            verbose=True
        )
        if result.get("success"):
            enhanced_results.append(result)

    # 3. 生成测试 - 直接使用增强检索的结果
    if enhanced_results:
        print("\n" + "=" * 70)
        print("3. 生成测试 (使用增强检索结果)")
        print("=" * 70)

        for result in enhanced_results:
            original_question = result.get("question", "")
            rewritten_query = result.get("rewritten_query", original_question)
            results = result.get("results", [])

            print(f"\n原始问题: {original_question}")
            print(f"改写后问题: {rewritten_query}")
            print(f"检索结果数: {len(results)}")

            # 使用改写后的问题进行生成
            generation_result = test_generation_single(
                question=rewritten_query,
                results=results,
                verbose=True
            )

            if generation_result.get("success"):
                print(f"\n✓ 生成成功")
                print(f"  模型: {generation_result['model_info']['model_name']}")
            else:
                print(f"\n✗ 生成失败")
    else:
        print("\n⚠ 没有增强检索结果，跳过生成测试")


def test_enhanced_retrieval_single(
    question: str,
    index_name: str,
    recall_k: int = 10,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
    verbose: bool = True
) -> dict:
    """单个问题的增强检索测试"""
    import os
    from .search import enhanced_search_with_hybrid_and_rerank

    result = enhanced_search_with_hybrid_and_rerank(
        question=question,
        index_name=index_name,
        recall_k=recall_k,
        top_k=top_k,
        keyword_weight=float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4")),
        vector_weight=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")),
        enable_rerank=os.getenv("ENABLE_RERANK", "true").lower() == "true",
        enable_query_rewrite=os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true",
        similarity_threshold=similarity_threshold,
        rerank_type=os.getenv("RERANK_TYPE", "remote"),
        verbose=verbose
    )

    return result


def test_generation_single(question: str, results: List[Dict], verbose: bool = True) -> dict:
    """单个问题的生成测试"""
    from .generation import generate_answer

    result = generate_answer(
        question=question,
        results=results,
        history=None,
        template_name="detailed",
        model_type=None,
        verbose=verbose,
        preview_answer=True
    )

    return result