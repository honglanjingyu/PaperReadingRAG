# app/service/core/rag/search.py (确保导出所有需要的函数)

"""
RAG 搜索模块 - 只负责调用检索器
"""

import os
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

from app.service.core.embedding import get_embedding_service
from app.service.core.retrieval.parent_child_retriever import get_parent_child_retriever
from .cached_search import CachedSearchService

# 创建全局缓存实例
_cached_search = CachedSearchService()


def vectorize_user_question(
        question: str,
        model_type: str = None,
        verbose: bool = False
) -> dict:
    """用户问题向量化"""
    try:
        embedding_service = get_embedding_service()
        if model_type == 'local':
            embedding_service.switch_to_local()
        else:
            embedding_service.switch_to_remote()

        question_vector = embedding_service.generate_embedding(question)

        if question_vector is None:
            return {"success": False, "question": question, "error": "问题向量化失败"}

        return {
            "success": True,
            "question": question,
            "question_length": len(question),
            "vector": question_vector,
            "vector_dimension": len(question_vector),
            "model_type": model_type,
            "model_info": embedding_service.get_model_info()
        }
    except Exception as e:
        return {"success": False, "question": question, "error": str(e)}


def enhanced_search_with_hybrid_and_rerank(
        question: str,
        index_name: str = None,
        top_k: int = 5,
        recall_k: int = 10,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
        enable_rerank: bool = True,
        enable_query_rewrite: bool = True,
        similarity_threshold: float = 0.3,
        rerank_type: str = "auto",
        verbose: bool = False,
        use_cache: bool = True,
        user_level: str = None
) -> dict:
    """
    增强检索 - 使用父子查询策略

    内部调用 ParentChildRetriever 进行检索
    """
    if index_name is None:
        index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    # 召回数量至少为 top_k 的 2 倍
    actual_recall_k = max(recall_k, top_k * 2)

    retriever = get_parent_child_retriever()

    if use_cache:
        result = retriever.search_with_cache(
            question=question,
            index_name=index_name,
            top_k=top_k,
            recall_k=actual_recall_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold,
            enable_rerank=enable_rerank,
            rerank_type=rerank_type,
            enable_query_rewrite=enable_query_rewrite,
            user_level=user_level,
            cache_ttl=300
        )
    else:
        result = retriever.search_with_query_rewrite(
            question=question,
            index_name=index_name,
            top_k=top_k,
            recall_k=actual_recall_k,
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold,
            enable_rerank=enable_rerank,
            rerank_type=rerank_type,
            enable_query_rewrite=enable_query_rewrite,
            user_level=user_level
        )

    # 添加额外信息
    result["index_name"] = index_name
    result["keyword_weight"] = keyword_weight
    result["vector_weight"] = vector_weight
    result["rerank_type"] = rerank_type
    result["rerank_model"] = os.getenv("RERANK_MODEL", "gte-rerank")

    return result


def search_similar_documents(
        question: str,
        VECTOR_INDEX_NAME: str = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        model_type: str = None,
        verbose: bool = False
) -> dict:
    """相似度搜索 - 兼容接口"""
    return enhanced_search_with_hybrid_and_rerank(
        question=question,
        index_name=VECTOR_INDEX_NAME,
        top_k=top_k,
        recall_k=top_k * 2,
        similarity_threshold=similarity_threshold,
        verbose=verbose
    )


# 导出所有函数
__all__ = [
    'vectorize_user_question',
    'search_similar_documents',
    'enhanced_search_with_hybrid_and_rerank'
]