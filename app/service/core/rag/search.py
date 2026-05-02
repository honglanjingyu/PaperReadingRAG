# app/service/core/rag/search.py
"""
RAG 搜索模块
包含：用户问题向量化 -> 相似度搜索 -> 增强检索
"""

import os
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

from app.service.core.embedding import VectorizationService, get_embedding_manager
from app.service.core.vector_store import get_vector_search_service


def vectorize_user_question(
    question: str,
    model_type: str = None,
    verbose: bool = True
) -> dict:
    """
    用户问题向量化

    Args:
        question: 用户问题文本
        model_type: 模型类型 ('remote' 或 'local')
        verbose: 是否打印详细信息

    Returns:
        dict: 包含问题文本、向量、向量维度、模型信息的结果字典
    """
    if verbose:
        print("\n[8/12] 接收用户问题...")
        print(f"\n用户问题: {question}")
        print(f"问题长度: {len(question)} 字符")

    if verbose:
        print("\n问题向量化...")

    try:
        if model_type is None:
            model_type = os.getenv("EMBEDDING_TYPE", "remote")

        vec_service = VectorizationService(model_type)
        question_vector = vec_service.manager.generate_embedding(question)

        if question_vector is None:
            print(f"  问题向量化失败")
            return None

        if verbose:
            print(f"  问题向量化成功")
            print(f"  向量维度: {len(question_vector)}")
            model_info = vec_service.get_model_info()
            print(f"  模型类型: {model_info.get('type', 'unknown')}")
            print(f"  模型名称: {model_info.get('model_name', 'unknown')}")

        return {
            "success": True,
            "question": question,
            "question_length": len(question),
            "vector": question_vector,
            "vector_dimension": len(question_vector),
            "model_type": model_type,
            "model_info": vec_service.get_model_info()
        }

    except Exception as e:
        if verbose:
            print(f"  问题向量化失败: {e}")
        return {"success": False, "question": question, "error": str(e)}


def search_similar_documents(
    question: str,
    es_index_name: str = None,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    model_type: str = None,
    verbose: bool = True
) -> dict:
    """
    相似度搜索：在向量数据库中召回最相关的 Top-K 个文档块
    """
    if verbose:
        print(f"\n[8/12] 用户问题: {question}")
        print(f"问题长度: {len(question)} 字符")
        print("\n问题向量化...")

    try:
        if model_type is None:
            model_type = os.getenv("EMBEDDING_TYPE", "remote")

        vec_service = VectorizationService(model_type)
        question_vector = vec_service.manager.generate_embedding(question)

        if question_vector is None:
            return {"success": False, "question": question, "error": "问题向量化失败"}

        if verbose:
            print(f"  向量维度: {len(question_vector)}")

    except Exception as e:
        if verbose:
            print(f"  向量化失败: {e}")
        return {"success": False, "question": question, "error": str(e)}

    if verbose:
        print(f"\n[9/12] 相似度搜索 (Top-K={top_k})...")

    try:
        if es_index_name is None:
            es_index_name = os.getenv("ES_INDEX_NAME", "rag_documents")

        search_service = get_vector_search_service()
        results = search_service.similarity_search(
            query_vector=question_vector,
            index_name=es_index_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        if verbose:
            print(f"  索引名称: {es_index_name}")
            print(f"  召回数量: {len(results)}/{top_k}")

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "score": result.get("_score", 0),
                "content": result.get("content_with_weight", result.get("content", "")),
                "document_name": result.get("docnm", result.get("docnm_kwd", "")),
                "chunk_id": result.get("_id", result.get("id", "")),
                "metadata": {k: v for k, v in result.items()
                           if k not in ["content", "content_with_weight", "_score", "_id", "id"]}
            })

        if verbose and formatted_results:
            print("\n" + "-" * 70)
            print("搜索结果详情:")
            print("-" * 70)
            for res in formatted_results:
                print(f"\n  [排名 {res['rank']}] 相似度: {res['score']:.4f}")
                print(f"  文档: {res['document_name']}")
                content_preview = res['content'][:200].replace('\n', ' ')
                print(f"  内容预览: {content_preview}...")

        return {
            "success": True,
            "question": question,
            "query_vector": question_vector,
            "vector_dimension": len(question_vector),
            "model_type": model_type,
            "index_name": es_index_name,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "total_recalled": len(results),
            "results": formatted_results
        }

    except Exception as e:
        if verbose:
            print(f"  相似度搜索失败: {e}")
        return {"success": False, "question": question, "error": str(e)}


# app/service/core/rag/search.py
# 修改 enhanced_search_with_hybrid_and_rerank 函数

def enhanced_search_with_hybrid_and_rerank(
        question: str,
        index_name: str = None,
        top_k: int = 5,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        enable_rerank: bool = True,
        enable_query_rewrite: bool = True,
        similarity_threshold: float = 0.3,
        rerank_type: str = "auto",  # auto, api, local, vector
        verbose: bool = True
) -> dict:
    """
    增强检索：混合检索 + 重排序 + Query改写
    """
    if verbose:
        print("\n" + "=" * 70)
        print("增强检索模式 (混合检索 + 重排序 + Query改写)")
        print("=" * 70)
        print(f"\n原始问题: {question}")

    if index_name is None:
        index_name = os.getenv("ES_INDEX_NAME", "rag_documents")

    # Query改写
    rewritten_query = question
    sub_queries = [question]
    if enable_query_rewrite:
        if verbose:
            print("\n[10/12] Query改写...")
        try:
            from app.service.core.retrieval import QueryRewriter
            rewriter = QueryRewriter()
            rewritten_query = rewriter.rewrite(question, strategy='synonym')
            sub_queries = rewriter.generate_sub_queries(question, max_queries=3)
            if verbose:
                print(f"  改写后问题: {rewritten_query}")
                print(f"  子查询: {sub_queries}")
        except Exception as e:
            if verbose:
                print(f"  Query改写失败: {e}")

    # 检查索引
    if verbose:
        print("\n检查索引...")

    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(index_name)

        if not index_exists:
            return {"success": False, "error": f"索引 '{index_name}' 不存在，请先处理文档"}

        doc_count = search_service.es_store.get_document_count(index_name)
        if verbose:
            print(f"  索引名称: {index_name}")
            print(f"  文档块数量: {doc_count}")

        if doc_count == 0:
            return {"success": False, "error": f"索引 '{index_name}' 为空"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    # 混合检索
    if verbose:
        print("\n混合检索...")
        print(f"  关键词权重: {keyword_weight}")
        print(f"  向量权重: {vector_weight}")

    try:
        from app.service.core.retrieval import HybridRetriever
        hybrid_retriever = HybridRetriever()
        hybrid_results = hybrid_retriever.hybrid_search(
            query=rewritten_query,
            index_name=index_name,
            top_k=top_k * 2,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold
        )
        if verbose:
            print(f"  混合检索召回: {len(hybrid_results)} 个块")
    except Exception as e:
        if verbose:
            print(f"  混合检索失败: {e}，回退到向量检索")
        from app.service.core.embedding import get_embedding_manager
        embedding_manager = get_embedding_manager()
        query_vector = embedding_manager.generate_embedding(rewritten_query)
        if query_vector:
            from app.service.core.vector_store import get_vector_search_service
            search_service = get_vector_search_service()
            hybrid_results = search_service.similarity_search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k * 2,
                similarity_threshold=similarity_threshold
            )
        else:
            hybrid_results = []

    if enable_rerank and hybrid_results:
        if verbose:
            print("\n[11/12] 重排序...")
            print(f"  重排序类型: {rerank_type}")
            # 显示配置的模型信息
            rerank_model = os.getenv("RERANK_MODEL", "gte-rerank")
            print(f"  Rerank模型: {rerank_model}")
            print(f"  Rerank API: {os.getenv('RERANK_API_URL', '未配置')}")

        try:
            from app.service.core.retrieval import Reranker
            reranker = Reranker(api_type=rerank_type)
            final_results = reranker.rerank(
                query=rewritten_query,
                documents=hybrid_results,
                top_k=top_k
            )

            if verbose:
                print(f"  重排序完成: {len(final_results)} 个块")
                # 显示实际使用的重排序方式
                if final_results and len(final_results) > 0:
                    source = final_results[0].get('rerank_source', 'unknown')
                    if source == 'dashscope_http':
                        print(f"  实际使用: DashScope HTTP API ({rerank_model})")
                    elif source == 'cross_encoder':
                        print(f"  实际使用: 本地 Cross-Encoder")
                    elif source == 'vector_similarity':
                        print(f"  实际使用: 向量相似度")
                    else:
                        print(f"  实际使用: {source}")
        except Exception as e:
            if verbose:
                print(f"  重排序失败: {e}，使用原始排序")
            final_results = hybrid_results[:top_k]
    else:
        final_results = hybrid_results[:top_k]
        if verbose and not enable_rerank:
            print("\n[10/12] 重排序已跳过 (enable_rerank=False)")

    formatted_results = []
    for i, result in enumerate(final_results, 1):
        formatted_results.append({
            "rank": i,
            "score": result.get("final_score", result.get("_score", result.get("rerank_score", 0))),
            "vector_score": result.get("vector_score", 0),
            "keyword_score": result.get("keyword_score", 0),
            "rerank_score": result.get("rerank_score", 0),
            "rerank_source": result.get("rerank_source", "unknown"),
            "content": result.get("content_with_weight", result.get("content", "")),
            "document_name": result.get("docnm", result.get("docnm_kwd", "")),
            "chunk_id": result.get("_id", result.get("id", "")),
            "search_type": result.get("_search_types", ["unknown"])
        })

    if verbose and formatted_results:
        print("\n增强检索结果详情:")
        print("-" * 70)
        for res in formatted_results:
            print(f"\n  [排名 {res['rank']}]")
            print(f"  综合分数: {res['score']:.4f}")
            if res.get('rerank_source'):
                print(f"  重排序来源: {res['rerank_source']}")
            print(f"  文档: {res['document_name']}")
            content_preview = res['content'][:200].replace('\n', ' ')
            print(f"  内容预览: {content_preview}...")

    return {
        "success": True,
        "question": question,
        "rewritten_query": rewritten_query if enable_query_rewrite else None,
        "index_name": index_name,
        "top_k": top_k,
        "keyword_weight": keyword_weight,
        "vector_weight": vector_weight,
        "enable_rerank": enable_rerank,
        "enable_query_rewrite": enable_query_rewrite,
        "rerank_type": rerank_type,
        "rerank_model": os.getenv("RERANK_MODEL", "gte-rerank"),
        "total_recalled": len(hybrid_results),
        "total_returned": len(formatted_results),
        "results": formatted_results
    }


def test_user_question_vectorization(questions: List[str] = None):
    """测试用户问题向量化功能"""
    from .utils import TEST_QUESTIONS
    if questions is None:
        questions = TEST_QUESTIONS

    print("\n默认测试问题:")
    for q in questions[:3]:
        print(f"  - {q}")
    if len(questions) > 3:
        print(f"  ... 共 {len(questions)} 个问题")

    results = []
    for question in questions:
        result = vectorize_user_question(question=question, verbose=False)
        if result and result.get("success"):
            print(f"\n✓ 问题: {question}")
            print(f"  向量维度: {result['vector_dimension']}")
            results.append(result)
        else:
            print(f"\n✗ 问题: {question}")
            print(f"  失败: {result.get('error', '未知错误') if result else '未知错误'}")
    return results


def test_similarity_search(questions: List[str] = None):
    """测试相似度搜索功能"""
    from .utils import TEST_QUESTIONS
    if questions is None:
        questions = TEST_QUESTIONS

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    top_k = int(os.getenv("SIMILARITY_TOP_K", "5"))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    print(f"\n配置信息:")
    print(f"  索引名称: {index_name}")
    print(f"  Top-K: {top_k}")
    print(f"  相似度阈值: {similarity_threshold}")

    all_results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"测试 {i}: {question}")
        print("=" * 70)

        result = search_similar_documents(
            question=question,
            es_index_name=index_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            verbose=True
        )

        if result.get("success"):
            print(f"\n✓ 测试 {i} 成功")
            print(f"  召回文档块数: {result['total_recalled']}")
            all_results.append(result)
        else:
            print(f"\n✗ 测试 {i} 失败: {result.get('error')}")

    return all_results


def test_enhanced_retrieval(questions: List[str] = None):
    """测试增强检索功能"""
    from .utils import TEST_QUESTIONS
    if questions is None:
        questions = TEST_QUESTIONS

    print("\n" + "=" * 70)
    print("测试增强检索功能")
    print("=" * 70)

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    top_k = int(os.getenv("SIMILARITY_TOP_K", "5"))

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"测试 {i}: {question}")
        print("=" * 70)

        modes = [
            ("向量检索", False, False, False),
            ("混合检索", True, False, False),
            ("混合检索+重排序", True, True, False),
            ("完整增强检索", True, True, True),
        ]

        for mode_name, use_hybrid, use_rerank, use_rewrite in modes:
            print(f"\n--- {mode_name} ---")
            result = enhanced_search_with_hybrid_and_rerank(
                question=question,
                index_name=index_name,
                top_k=top_k,
                keyword_weight=0.3 if use_hybrid else 1.0,
                vector_weight=0.7 if use_hybrid else 1.0,
                enable_rerank=use_rerank,
                enable_query_rewrite=use_rewrite,
                verbose=False
            )

            if result.get("success"):
                print(f"  召回数量: {result['total_returned']}")
                if result['results']:
                    best = result['results'][0]
                    print(f"  最佳匹配: {best['document_name']}")
                    print(f"  综合分数: {best['score']:.4f}")
            else:
                print(f"  失败: {result.get('error')}")


def compare_search_methods(questions: List[str] = None):
    """对比传统检索和增强检索的效果"""
    from .utils import TEST_QUESTIONS
    if questions is None:
        questions = TEST_QUESTIONS

    print("\n" + "=" * 70)
    print("传统检索 vs 增强检索 对比")
    print("=" * 70)

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    top_k = int(os.getenv("SIMILARITY_TOP_K", "5"))

    comparison_results = []

    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 50)

        from app.service.core.embedding import get_embedding_manager
        from app.service.core.vector_store import get_vector_search_service

        embedding_manager = get_embedding_manager()
        query_vector = embedding_manager.generate_embedding(question)

        if query_vector:
            search_service = get_vector_search_service()
            traditional_results = search_service.similarity_search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k,
                similarity_threshold=0.3
            )
            traditional_avg_score = sum(r.get('_score', 0) for r in traditional_results) / len(traditional_results) if traditional_results else 0
            traditional_count = len(traditional_results)
        else:
            traditional_avg_score = 0
            traditional_count = 0

        enhanced_result = enhanced_search_with_hybrid_and_rerank(
            question=question,
            index_name=index_name,
            top_k=top_k,
            keyword_weight=0.3,
            vector_weight=0.7,
            enable_rerank=True,
            enable_query_rewrite=True,
            verbose=False
        )

        enhanced_count = enhanced_result.get('total_returned', 0) if enhanced_result.get('success') else 0
        enhanced_avg_score = 0
        if enhanced_result.get('success') and enhanced_result['results']:
            enhanced_avg_score = sum(r['score'] for r in enhanced_result['results']) / len(enhanced_result['results'])

        print(f"  传统向量检索: 召回 {traditional_count} 块, 平均分数 {traditional_avg_score:.4f}")
        print(f"  增强检索: 召回 {enhanced_count} 块, 平均分数 {enhanced_avg_score:.4f}")

        comparison_results.append({
            "question": question,
            "traditional": {"count": traditional_count, "avg_score": traditional_avg_score},
            "enhanced": {"count": enhanced_count, "avg_score": enhanced_avg_score}
        })

    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)

    avg_traditional_count = sum(r["traditional"]["count"] for r in comparison_results) / len(comparison_results)
    avg_enhanced_count = sum(r["enhanced"]["count"] for r in comparison_results) / len(comparison_results)
    avg_traditional_score = sum(r["traditional"]["avg_score"] for r in comparison_results) / len(comparison_results)
    avg_enhanced_score = sum(r["enhanced"]["avg_score"] for r in comparison_results) / len(comparison_results)

    print(f"\n平均召回数量: 传统={avg_traditional_count:.1f} | 增强={avg_enhanced_count:.1f} | 提升={avg_enhanced_count - avg_traditional_count:.1f}")
    print(f"平均相似度: 传统={avg_traditional_score:.4f} | 增强={avg_enhanced_score:.4f} | 提升={avg_enhanced_score - avg_traditional_score:.4f}")

    return comparison_results