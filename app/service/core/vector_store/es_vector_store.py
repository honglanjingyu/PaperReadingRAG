# app/service/core/vector_store/es_vector_store.py
"""Elasticsearch 向量数据库存储实现"""

import xxhash
import os
import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class ESVectorStore:
    """Elasticsearch 向量数据库存储"""

    def __init__(self, es_host: str = None, es_user: str = None, es_password: str = None):
        """
        初始化 ES 连接

        Args:
            es_host: ES 主机地址 (例如: http://localhost:9200)
            es_user: ES 用户名
            es_password: ES 密码
        """
        self.es_host = es_host
        self.es_user = es_user or ""
        self.es_password = es_password or ""

        if not self.es_host:
            raise ValueError("es_host 不能为空，请提供 Elasticsearch 连接地址")

        logger.info(f"连接 Elasticsearch: {self.es_host}")

        try:
            # 构建认证参数
            auth_params = {}
            if self.es_user and self.es_password:
                auth_params['basic_auth'] = (self.es_user, self.es_password)

            self.es = Elasticsearch(
                [self.es_host],
                verify_certs=False,
                request_timeout=60,
                **auth_params
            )

            # 测试连接
            info = self.es.info()
            logger.info(f"Elasticsearch 连接成功，版本: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Elasticsearch 连接失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        try:
            if hasattr(self, 'es') and self.es:
                self.es.close()
                logger.info("Elasticsearch 连接已关闭")
        except Exception as e:
            logger.error(f"关闭 Elasticsearch 连接失败: {e}")

    def create_index(self, index_name: str, vector_dim: int = None, **kwargs) -> bool:
        """创建索引"""
        if vector_dim is None:
            vector_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

        # 检查索引是否存在
        if self.es.indices.exists(index=index_name):
            logger.info(f"索引已存在: {index_name}")
            return True

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
            return True
        except Exception as e:
            logger.error(f"索引创建失败: {e}")
            return False

    def insert(self, documents: List[Dict[str, Any]], index_name: str) -> int:
        """批量插入文档"""
        if not documents:
            return 0

        # 确保索引存在
        vector_dim = None
        for doc in documents:
            for key in doc:
                if key.endswith("_vec") and isinstance(doc[key], list):
                    vector_dim = len(doc[key])
                    break
            if vector_dim:
                break

        if vector_dim:
            self.create_index(index_name, vector_dim)

        # 批量插入
        success_count = 0
        for doc in documents:
            doc_copy = doc.copy()
            doc_id = doc_copy.pop("id", None)

            if not doc_id:
                content = doc_copy.get("content", "") or doc_copy.get("content_with_weight", "")
                doc_id = xxhash.xxh64(content.encode("utf-8")).hexdigest()

            try:
                self.es.index(
                    index=index_name,
                    id=doc_id,
                    body=doc_copy,
                    refresh=True
                )
                success_count += 1
            except Exception as e:
                logger.error(f"插入文档失败: {e}")

        logger.info(f"批量插入完成: {success_count}/{len(documents)} 条成功")
        return success_count

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

    def delete_index(self, index_name: str) -> bool:
        """删除整个索引"""
        try:
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                logger.info(f"索引删除成功: {index_name}")
            return True
        except Exception as e:
            logger.error(f"索引删除失败: {e}")
            return False

    def search(self, query_vector: List[float], index_name: str, top_k: int = 5,
               filter_condition: Optional[Dict] = None, similarity_threshold: float = 0.5) -> List[Dict]:
        """向量相似度搜索"""
        try:
            if not self.es.indices.exists(index=index_name):
                logger.warning(f"索引不存在: {index_name}")
                return []

            vector_dim = len(query_vector)
            vector_field = f"q_{vector_dim}_vec"

            # 使用 script_score 查询
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }

            body = {
                "size": top_k,
                "query": script_query
            }

            response = self.es.search(index=index_name, body=body)

            results = []
            for hit in response["hits"]["hits"]:
                score = hit.get("_score", 0) - 1.0
                if score >= similarity_threshold:
                    result = hit["_source"]
                    result["_score"] = score
                    result["_id"] = hit["_id"]
                    results.append(result)

            logger.info(f"向量搜索完成: 召回 {len(results)} 个文档")
            return results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def index_exists(self, index_name: str) -> bool:
        """检查索引是否存在"""
        try:
            return self.es.indices.exists(index=index_name)
        except Exception as e:
            logger.error(f"检查索引存在失败: {e}")
            return False

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