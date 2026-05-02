# app/service/core/rag/utils.py
"""
RAG 工具函数和测试
"""

import os
from typing import List, Dict

# 全局测试问题列表
TEST_QUESTIONS = [
    "世运电子的主要业务是什么？",
    # "公司2023年中报的营收情况如何？",
    # "请分析世运电子的盈利能力",
    # "公司的主要客户有哪些？",
    # "世运电子的竞争优势是什么？",
    # "世运电子的盈利能力怎么样？",
]


# 向后兼容的分块器类
class RecursiveChunker:
    """向后兼容的分块器类"""
    def __init__(self, chunk_token_num: int = 256):
        from app.service.core.chunking import RecursiveChunkerSimple
        self.chunker = RecursiveChunkerSimple(chunk_token_num=chunk_token_num)

    def chunk(self, text: str) -> List[str]:
        return self.chunker.chunk(text)


# app/service/core/rag/utils.py

def run_all_tests():
    """运行所有测试"""
    from .search import (
        test_user_question_vectorization,
        test_similarity_search,
        test_enhanced_retrieval,
    )
    from .generation import test_llm_generation

    print("\n" + "=" * 70)
    print("运行所有测试")
    print("=" * 70)

    # 1. 用户问题向量化测试
    print("\n" + "=" * 70)
    print("1. 用户问题向量化测试")
    print("=" * 70)
    test_user_question_vectorization()

    # 2. 相似度搜索测试
    print("\n" + "=" * 70)
    print("2. 相似度搜索测试")
    print("=" * 70)

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(index_name)

        if index_exists and search_service.es_store.get_document_count(index_name) > 0:
            test_similarity_search()
        else:
            print(f"\n⚠ 索引 '{index_name}' 不存在或为空，跳过相似度搜索测试")
    except Exception as e:
        print(f"\n检查索引时出错: {e}")

    # 3. 增强检索测试 - 改为 verbose=True
    print("\n" + "=" * 70)
    print("3. 增强检索测试")
    print("=" * 70)

    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(index_name)

        if index_exists and search_service.es_store.get_document_count(index_name) > 0:
            # 直接调用 enhanced_search_with_hybrid_and_rerank 而不是 test_enhanced_retrieval
            from .search import enhanced_search_with_hybrid_and_rerank
            for question in TEST_QUESTIONS[:1]:
                print(f"\n详细测试问题: {question}")
                result = enhanced_search_with_hybrid_and_rerank(
                    question=question,
                    index_name=index_name,
                    top_k=int(os.getenv("SIMILARITY_TOP_K", "5")),
                    keyword_weight=float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4")),
                    vector_weight=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")),
                    enable_rerank=True,
                    enable_query_rewrite=True,
                    similarity_threshold=0.3,
                    rerank_type=os.getenv("RERANK_TYPE", "remote"),
                    verbose=True  # 显示详细信息
                )
        else:
            print(f"\n⚠ 索引 '{index_name}' 不存在或为空，跳过增强检索测试")
    except Exception as e:
        print(f"\n增强检索测试失败: {e}")

    # 4. 生成测试
    run_generation_tests()


def run_generation_tests():
    """运行生成测试"""
    from .generation import test_llm_generation

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")

    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(index_name)

        if not index_exists or search_service.es_store.get_document_count(index_name) == 0:
            print(f"\n⚠ 索引 '{index_name}' 不存在或为空，跳过生成测试")
            return

        test_llm_generation(TEST_QUESTIONS[:2], stream=False)

    except Exception as e:
        print(f"\n生成测试失败: {e}")