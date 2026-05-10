# app/service/core/rag/search.py - 清理后的版本

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
from .cached_search import CachedSearchService

# 创建全局缓存实例
_cached_search = CachedSearchService()

def vectorize_user_question(
        question: str,
        model_type: str = None,
        verbose: bool = False
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

    try:
        if model_type is None:
            model_type = os.getenv("EMBEDDING_TYPE", "remote")

        vec_service = VectorizationService(model_type)
        question_vector = vec_service.manager.generate_embedding(question)

        if question_vector is None:
            return {"success": False, "question": question, "error": "问题向量化失败"}

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
        return {"success": False, "question": question, "error": str(e)}


def search_similar_documents(
        question: str,
        VECTOR_INDEX_NAME: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        model_type: str = None,
        verbose: bool = False
) -> dict:
    """
    相似度搜索：在向量数据库中召回最相关的 Top-K 个文档块
    """

    try:
        if model_type is None:
            model_type = os.getenv("EMBEDDING_TYPE", "remote")

        vec_service = VectorizationService(model_type)
        question_vector = vec_service.manager.generate_embedding(question)

        if question_vector is None:
            return {"success": False, "question": question, "error": "问题向量化失败"}

    except Exception as e:
        return {"success": False, "question": question, "error": str(e)}

    try:
        if VECTOR_INDEX_NAME is None:
            VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        search_service = get_vector_search_service()
        results = search_service.similarity_search(
            query_vector=question_vector,
            index_name=VECTOR_INDEX_NAME,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # 确保结果按 _score 降序排序
        results = sorted(results, key=lambda x: x.get("_score", 0), reverse=True)

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

        return {
            "success": True,
            "question": question,
            "query_vector": question_vector,
            "vector_dimension": len(question_vector),
            "model_type": model_type,
            "index_name": VECTOR_INDEX_NAME,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "total_recalled": len(results),
            "results": formatted_results
        }

    except Exception as e:
        return {"success": False, "question": question, "error": str(e)}


def _enhanced_search_internal(
        question: str,
        index_name: str = None,
        top_k: int = 5,
        recall_k: int = 10,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        enable_rerank: bool = True,
        enable_query_rewrite: bool = True,
        similarity_threshold: float = 0.3,
        rerank_type: str = "auto",
        verbose: bool = False,
        user_level: str = None
) -> dict:
    """
    增强检索内部实现（不包含缓存）
    """
    # 健壮性检查
    if top_k > recall_k:
        recall_k = top_k

    if top_k <= 0:
        top_k = 1
    if recall_k <= 0:
        recall_k = 1

    if index_name is None:
        index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    # Query改写
    rewritten_query = question
    sub_queries = [question]

    if enable_query_rewrite:
        try:
            from app.service.core.retrieval import QueryRewriter
            rewriter = QueryRewriter()
            rewritten_query = rewriter.rewrite(question, strategy='synonym')
        except Exception as e:
            pass

    # 检查索引
    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.store.index_exists(index_name)

        if not index_exists:
            return {"success": False, "error": f"索引 '{index_name}' 不存在，请先处理文档"}

        doc_count = search_service.store.get_document_count(index_name)
        if doc_count == 0:
            return {"success": False, "error": f"索引 '{index_name}' 为空"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    # 相似度搜索（使用改写后的问题）
    try:
        from app.service.core.retrieval import HybridRetriever
        hybrid_retriever = HybridRetriever(use_jieba=True)

        hybrid_results = hybrid_retriever.hybrid_search(
            query=rewritten_query,
            index_name=index_name,
            top_k=recall_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold,
            verbose=False,
            user_level=user_level
        )

        hybrid_results.sort(key=lambda x: x.get('final_score', x.get('_score', 0)), reverse=True)

    except Exception as e:
        from app.service.core.embedding import get_embedding_manager
        embedding_manager = get_embedding_manager()
        query_vector = embedding_manager.generate_embedding(rewritten_query)
        if query_vector:
            from app.service.core.vector_store import get_vector_search_service
            search_service = get_vector_search_service()
            hybrid_results = search_service.similarity_search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=recall_k,
                similarity_threshold=similarity_threshold,
                user_level=user_level
            )
        else:
            hybrid_results = []

    # 重排序（使用改写后的问题）
    final_results = hybrid_results

    if enable_rerank and hybrid_results:
        try:
            from app.service.core.retrieval import Reranker
            reranker = Reranker(api_type=rerank_type)

            final_results = reranker.rerank(
                query=rewritten_query,
                documents=hybrid_results,
                top_k=top_k
            )

            if final_results:
                final_results.sort(key=lambda x: x.get('rerank_score', x.get('final_score', 0)), reverse=True)
                final_results = final_results[:top_k]

        except Exception as e:
            final_results = hybrid_results[:top_k]
            final_results.sort(key=lambda x: x.get('final_score', x.get('_score', 0)), reverse=True)
    else:
        final_results = hybrid_results[:top_k]
        if final_results:
            final_results.sort(key=lambda x: x.get('final_score', x.get('_score', 0)), reverse=True)

    # 格式化结果
    formatted_results = []
    for i, result in enumerate(final_results, 1):
        score = result.get('rerank_score',
                           result.get('final_score',
                                      result.get('_score', 0)))

        formatted_results.append({
            "rank": i,
            "score": score,
            "vector_score": result.get("vector_score", 0),
            "keyword_score": result.get("keyword_score", 0),
            "rerank_score": result.get("rerank_score", 0),
            "rerank_source": result.get("rerank_source", "unknown"),
            "content": result.get("content_with_weight", result.get("content", "")),
            "document_name": result.get("docnm", result.get("docnm_kwd", "")),
            "chunk_id": result.get("_id", result.get("id", "")),
            "search_type": result.get("_search_types", ["unknown"])
        })

    return {
        "success": True,
        "question": question,
        "rewritten_query": rewritten_query if enable_query_rewrite else None,
        "index_name": index_name,
        "top_k": top_k,
        "recall_k": recall_k,
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


def enhanced_search_with_hybrid_and_rerank(
        question: str,
        index_name: str = None,
        top_k: int = 5,
        recall_k: int = 10,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
        enable_rerank: bool = True,
        enable_query_rewrite: bool = True,
        similarity_threshold: float = 0.3,
        rerank_type: str = "auto",
        verbose: bool = False,
        use_cache: bool = True,
        user_level: str = None  # 新增参数
) -> dict:
    """增强检索（支持用户等级过滤）"""

    if not use_cache:
        return _enhanced_search_internal(
            question=question,
            index_name=index_name,
            top_k=top_k,
            recall_k=recall_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            enable_rerank=enable_rerank,
            enable_query_rewrite=enable_query_rewrite,
            similarity_threshold=similarity_threshold,
            rerank_type=rerank_type,
            verbose=verbose,
            user_level=user_level  # 传递用户等级
        )

    return _cached_search.search_with_cache(
        question=question,
        search_func=_enhanced_search_internal,
        top_k=top_k,
        recall_k=recall_k,
        similarity_threshold=similarity_threshold,
        enable_rerank=enable_rerank,
        enable_query_rewrite=enable_query_rewrite,
        index_name=index_name,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
        rerank_type=rerank_type,
        verbose=verbose,
        user_level=user_level  # 传递用户等级
    )

__all__ = [
    'vectorize_user_question',
    'search_similar_documents',
    'enhanced_search_with_hybrid_and_rerank'
]