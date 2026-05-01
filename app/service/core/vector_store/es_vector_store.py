# app/service/core/vector_store/es_vector_store.py

import xxhash
import datetime
import os
import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class ESVectorStore:
    """Elasticsearch 向量数据库存储 - 单节点优化版"""

    def __init__(self, es_host: str = None, es_user: str = "elastic", es_password: str = "infini_rag_flow"):
        """
        初始化 ES 连接

        Args:
            es_host: ES 主机地址，默认从环境变量读取
            es_user: ES 用户名
            es_password: ES 密码
        """
        self.es_host = es_host or os.getenv("ES_HOST", "http://localhost:9200")
        self.es_user = es_user
        self.es_password = es_password

        self.es = Elasticsearch(
            [self.es_host],
            basic_auth=(es_user, es_password),
            verify_certs=False,
            timeout=60
        )

        # 测试连接
        try:
            info = self.es.info()
            logger.info(f"Elasticsearch 连接成功: {self.es_host}, 版本: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Elasticsearch 连接失败: {e}")
            raise

    def create_index(self, index_name: str, vector_dim: int = 1536):
        """
        创建索引 - 单节点优化配置

        Args:
            index_name: 索引名称
            vector_dim: 向量维度
        """
        if self.es.indices.exists(index=index_name):
            logger.info(f"索引已存在: {index_name}")
            return

        # 索引映射配置
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s"
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "content_with_weight": {"type": "text"},
                    "docnm": {"type": "text"},
                    "docnm_kwd": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "kb_id": {"type": "keyword"},
                    "create_time": {"type": "date"},
                    "create_timestamp_flt": {"type": "float"},
                    "token_count": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    f"q_{vector_dim}_vec": {
                        "type": "dense_vector",
                        "dims": vector_dim,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }

        try:
            self.es.indices.create(index=index_name, body=mapping)
            logger.info(f"索引创建成功: {index_name} (维度:{vector_dim})")
        except Exception as e:
            logger.error(f"索引创建失败: {e}")
            raise

    def insert(self, documents: List[Dict[str, Any]], index_name: str) -> int:
        """
        批量插入文档

        Args:
            documents: 文档列表（每个文档应包含 id 字段）
            index_name: 索引名称

        Returns:
            成功插入的文档数量
        """
        if not documents:
            return 0

        operations = []
        for doc in documents:
            doc_copy = doc.copy()
            doc_id = doc_copy.pop("id", None)

            if not doc_id:
                content = doc_copy.get("content", "") or doc_copy.get("content_with_weight", "")
                doc_id = xxhash.xxh64(content.encode("utf-8")).hexdigest()

            operations.append({"index": {"_index": index_name, "_id": doc_id}})
            operations.append(doc_copy)

        try:
            response = self.es.bulk(operations=operations, refresh=True, timeout="60s")

            if response.get("errors"):
                failed_count = 0
                for item in response.get("items", []):
                    if "error" in item.get("index", {}):
                        failed_count += 1
                return len(documents) - failed_count
            else:
                logger.info(f"批量插入成功: {len(documents)} 条文档 -> {index_name}")
                return len(documents)

        except Exception as e:
            logger.error(f"ES 批量插入失败: {e}")
            return 0

    def delete(self, index_name: str, condition: Dict[str, Any]) -> int:
        """删除符合条件的文档"""
        try:
            if not self.es.indices.exists(index=index_name):
                return 0

            query = {
                "query": {
                    "bool": {
                        "must": []
                    }
                }
            }

            for field, value in condition.items():
                if isinstance(value, list):
                    query["query"]["bool"]["must"].append({"terms": {field: value}})
                else:
                    query["query"]["bool"]["must"].append({"term": {field: value}})

            response = self.es.delete_by_query(index=index_name, body=query, refresh=True)
            deleted = response.get("deleted", 0)
            logger.info(f"删除文档: {deleted} 条")
            return deleted

        except NotFoundError:
            return 0
        except Exception as e:
            logger.error(f"ES 删除失败: {e}")
            return 0

    def delete_index(self, index_name: str):
        """删除整个索引"""
        try:
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                logger.info(f"索引删除成功: {index_name}")
        except Exception as e:
            logger.error(f"索引删除失败: {e}")

    def search(self, query_vector: List[float], index_name: str, top_k: int = 5,
               filter_condition: Optional[Dict] = None, similarity_threshold: float = 0.5) -> List[Dict]:
        """向量相似度搜索"""
        try:
            if not self.es.indices.exists(index=index_name):
                return []

            vector_dim = len(query_vector)
            vector_field = f"q_{vector_dim}_vec"

            knn_query = {
                "field": vector_field,
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 2
            }

            if filter_condition:
                knn_query["filter"] = filter_condition

            body = {
                "knn": knn_query,
                "_source": True,
                "size": top_k
            }

            response = self.es.search(index=index_name, body=body)

            results = []
            for hit in response["hits"]["hits"]:
                score = hit.get("_score", 0)
                if score >= similarity_threshold:
                    result = hit["_source"]
                    result["_score"] = score
                    result["_id"] = hit["_id"]
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def get_document_count(self, index_name: str) -> int:
        """获取索引中的文档数量"""
        try:
            if not self.es.indices.exists(index=index_name):
                return 0
            response = self.es.count(index=index_name)
            return response.get("count", 0)
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    def index_exists(self, index_name: str) -> bool:
        """检查索引是否存在"""
        try:
            return self.es.indices.exists(index=index_name)
        except Exception as e:
            logger.error(f"检查索引存在失败: {e}")
            return False