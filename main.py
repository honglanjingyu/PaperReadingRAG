# RAG4.py
"""
RAG 完整流程入口
处理顺序：
RAG1 流程: 数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗"
RAG2 流程: 智能分块 -> 向量化 -> 向量存储"
RAG3 流程: 用户问题向量化 -> 相似度搜索 -> 增强搜索"
RAG4 流程: 上下文构造 -> 推理生成"
"""

import os
import sys
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 导入 RAG 核心模块
from app.service.core.rag import (
    # 类型和常量
    TEST_QUESTIONS,

    # 文档处理
    process_document,
    parse_only,
    chunk_document,
    get_processing_stats,

    # 搜索功能
    vectorize_user_question,
    search_similar_documents,
    enhanced_search_with_hybrid_and_rerank,
    test_user_question_vectorization,
    test_similarity_search,
    test_enhanced_retrieval,
    compare_search_methods,

    # 生成功能
    generate_answer,
    generate_answer_stream,

    # 测试
    run_all_tests,
)

# 导出所有公共接口
__all__ = [
    'TEST_QUESTIONS',
    'process_document',
    'parse_only',
    'chunk_document',
    'get_processing_stats',
    'vectorize_user_question',
    'search_similar_documents',
    'enhanced_search_with_hybrid_and_rerank',
    'generate_answer',
    'generate_answer_stream',
    'test_user_question_vectorization',
    'test_similarity_search',
    'test_enhanced_retrieval',
    'compare_search_methods',
    'run_all_tests',
]


# main.py 中的 main 函数简化

def main():
    """主入口函数"""
    print("\n处理流程:")
    print("  RAG1 流程: 数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗")
    print("  RAG2 流程: 智能分块 -> 向量化 -> 向量存储")
    print("  RAG3 流程: 用户问题向量化 -> 相似度搜索 -> 增强搜索")
    print("  RAG4 流程: 上下文构造 -> 推理生成")
    print("=" * 70)

    # 从环境变量读取配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
    ENABLE_VECTORIZATION = os.getenv("ENABLE_VECTORIZATION", "true").lower() == "true"
    ENABLE_STORAGE = True
    MODEL_TYPE = os.getenv("EMBEDDING_TYPE", "remote")
    INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    print(f"\n配置信息:")
    print(f"  分块大小: {CHUNK_SIZE} tokens")
    print(f"  向量化: {'启用' if ENABLE_VECTORIZATION else '禁用'}")
    print(f"  存储: {'启用' if ENABLE_STORAGE else '禁用'} (默认启用)")
    if ENABLE_VECTORIZATION:
        print(f"  模型类型: {MODEL_TYPE}")
    print(f"  索引名称: {INDEX_NAME}")

    print(f"\n全局测试问题列表 ({len(TEST_QUESTIONS)} 个):")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"  {i}. {q}")

    # 测试 PDF 文件
    pdf_file = "【兴证电子】世运电路2023中报点评.pdf"

    if os.path.exists(pdf_file):
        result = process_document(
            file_path=pdf_file,
            chunk_size=CHUNK_SIZE,
            enable_vectorization=ENABLE_VECTORIZATION,
            enable_storage=ENABLE_STORAGE,
            model_type=MODEL_TYPE,
            from_page=0,
            to_page=5,
            verbose=True
        )

        print(f"\n处理结果摘要:")
        print(f"  生成块数: {len(result)}")

        if result and hasattr(result[0], 'vector') and result[0].vector:
            vectorized = [c for c in result if c.vector]
            print(f"  向量化块数: {len(vectorized)}")
            print(f"  向量维度: {len(vectorized[0].vector)}")

            print(f"\n前3个块预览:")
            for i, chunk in enumerate(result[:3]):
                preview = chunk.content[:150].replace('\n', ' ')
                token_count = chunk.token_count if hasattr(chunk, 'token_count') else 0
                print(f"\n  块 {i + 1} (Token: {token_count}):")
                print(f"    {preview}...")

    else:
        print(f"\n测试文件不存在: {pdf_file}")
        print("请将 PDF 文件放在当前目录下")

    # 运行测试 - 已优化避免重复
    run_all_tests()

if __name__ == "__main__":
    main()