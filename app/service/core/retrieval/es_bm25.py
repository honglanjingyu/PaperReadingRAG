# app/service/core/retrieval/es_bm25.py
"""
Elasticsearch BM25 检索器
"""

import os
import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)

try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


class ESBM25Retriever:
    """ES BM25 检索器"""

    def __init__(self, es_host: str = None, es_port: int = None):
        self.es_host = es_host or os.getenv("ES_HOST", "localhost")
        self.es_port = es_port or int(os.getenv("ES_PORT", "9200"))
        self._client = None
        self._index_prefix = "rag_bm25_"
        self._init_client()

    def _init_client(self):
        try:
            self._client = Elasticsearch([f"http://{self.es_host}:{self.es_port}"],
                                         request_timeout=30, max_retries=3)
            if self._client.ping():
                logger.info(f"ES 连接成功: {self.es_host}:{self.es_port}")
            else:
                self._client = None
        except Exception as e:
            logger.error(f"ES 连接失败: {e}")
            self._client = None

    def _get_index_name(self, index_name: str) -> str:
        # 确保 index_name 是字符串
        if not isinstance(index_name, str):
            index_name = str(index_name)
        safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in index_name)
        return f"{self._index_prefix}{safe_name}"

    def index_documents(self, documents: List[Dict], index_name: str, user_level: str = "normal") -> int:
        """批量索引文档到 ES"""
        if not self.is_available():
            logger.warning("ES 不可用，跳过索引")
            return 0

        # 确保 index_name 是字符串
        if not isinstance(index_name, str):
            index_name = str(index_name)

        es_index = self._get_index_name(index_name)

        # 创建索引（如果不存在）
        if not self._client.indices.exists(index=es_index):
            mapping = {
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "content": {"type": "text", "analyzer": "standard"},
                        "content_with_weight": {"type": "text", "analyzer": "standard"},
                        "document_name": {"type": "keyword"},
                        "docnm": {"type": "keyword"},
                        "user_level": {"type": "keyword"},
                        "kb_id": {"type": "keyword"},
                        "token_count": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                        "parent_id": {"type": "keyword"},
                        "parent_content": {"type": "text"},
                        "chunk_type": {"type": "keyword"},
                        "created_at": {"type": "date"}
                    }
                }
            }
            self._client.indices.create(index=es_index, body=mapping)
            logger.info(f"ES 索引创建成功: {es_index}")

        # 批量插入
        actions = []
        for doc in documents:
            action = {
                "_op_type": "index",
                "_index": es_index,
                "_id": doc.get("id"),
                "_source": doc
            }
            actions.append(action)

        if actions:
            success, failed = bulk(self._client, actions, stats_only=True, raise_on_error=False)
            logger.info(f"ES 索引完成: 成功 {success}, 失败 {failed}")
            return success

        return 0

    def get_document_count(self, index_name: str) -> int:
        """获取索引中的文档数量"""
        if not isinstance(index_name, str):
            index_name = str(index_name)
        es_index = self._get_index_name(index_name)
        if not self._client or not self._client.indices.exists(index=es_index):
            return 0

        try:
            result = self._client.count(index=es_index)
            return result.get("count", 0)
        except Exception as e:
            logger.error(f"获取 ES 文档数量失败: {e}")
            return 0

    def search(self, query: str, index_name: str, top_k: int = 10,
               min_score: float = 0.1, user_level: str = None) -> List[Dict]:
        """BM25 搜索"""
        # 确保 index_name 是字符串
        if not isinstance(index_name, str):
            index_name = str(index_name)

        es_index = self._get_index_name(index_name)
        if not self._client or not self._client.indices.exists(index=es_index):
            return []

        # 构建过滤条件
        filter_queries = []
        if user_level:
            level_priority = {"normal": 1, "admin": 2, "owner": 3}
            current = level_priority.get(user_level, 1)
            allowed = [l for l, p in level_priority.items() if p <= current]
            filter_queries.append({"terms": {"user_level": allowed}})

        search_body = {
            "query": {
                "bool": {
                    "must": [{"multi_match": {"query": query, "fields": ["content^2", "content_with_weight^1.5"]}}],
                    "filter": filter_queries if filter_queries else None
                }
            },
            "size": top_k,
            "min_score": min_score
        }

        try:
            response = self._client.search(index=es_index, body=search_body)
            results = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                results.append({
                    "_id": hit.get("_id"),
                    "_score": hit.get("_score", 0),
                    "content": source.get("content", ""),
                    "content_with_weight": source.get("content_with_weight", ""),
                    "document_name": source.get("document_name", source.get("docnm", "")),
                    "docnm": source.get("docnm", ""),
                    "user_level": source.get("user_level", "normal"),
                    "_search_type": "es_bm25"
                })
            return results
        except Exception as e:
            logger.error(f"ES 搜索失败: {e}")
            return []

    def is_available(self) -> bool:
        return self._client is not None and self._client.ping()


_es_bm25 = None


def get_es_bm25_retriever() -> ESBM25Retriever:
    global _es_bm25
    if _es_bm25 is None:
        _es_bm25 = ESBM25Retriever()
    return _es_bm25


__all__ = ['ESBM25Retriever', 'get_es_bm25_retriever']