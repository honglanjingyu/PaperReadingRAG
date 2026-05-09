# app/service/core/retrieval/es_bm25_retriever.py
"""
Elasticsearch BM25 检索器 - 使用 ES 内置 BM25 提高效率
ES 的 BM25 是原生实现的，比 rank_bm25 快 10-100 倍
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from elasticsearch import Elasticsearch, exceptions as es_exceptions

logger = logging.getLogger(__name__)

# 尝试导入 jieba
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba 未安装，使用简单分词")


class ESBM25Retriever:
    """
    Elasticsearch BM25 检索器

    优势：
    1. ES 原生 BM25 实现，速度快
    2. 支持增量更新，无需每次重建整个索引
    3. 支持高亮、聚合、过滤等高级功能
    4. 支持同义词扩展
    """

    def __init__(
            self,
            es_host: str = None,
            es_port: int = None,
            es_user: str = None,
            es_password: str = None,
            use_jieba: bool = True,
            use_synonyms: bool = True,
            bm25_k1: float = 1.2,
            bm25_b: float = 0.75
    ):
        """
        初始化 ES BM25 检索器

        Args:
            es_host: Elasticsearch 主机地址
            es_port: Elasticsearch 端口
            es_user: 用户名
            es_password: 密码
            use_jieba: 是否使用 jieba 分词（用于中文）
            use_synonyms: 是否使用同义词扩展
            bm25_k1: BM25 k1 参数（默认 1.2）
            bm25_b: BM25 b 参数（默认 0.75）
        """
        # 读取 ES 配置
        self.es_host = es_host or os.getenv("ES_HOST", "localhost")
        self.es_port = es_port or int(os.getenv("ES_PORT", "9200"))
        self.es_user = es_user or os.getenv("ES_USER", "")
        self.es_password = es_password or os.getenv("ES_PASSWORD", "")

        self.use_jieba = use_jieba and JIEBA_AVAILABLE
        self.use_synonyms = use_synonyms
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self._client: Optional[Elasticsearch] = None
        self._index_prefix = "rag_bm25_"

        # 加载同义词
        self._synonyms = {}
        if use_synonyms:
            self._load_synonyms()

        self._init_client()

    def _load_synonyms(self):
        """加载同义词表"""
        try:
            from app.service.synonymlist.loader import get_synonym_loader
            loader = get_synonym_loader()
            self._synonyms = loader.get_all_synonyms()
            logger.info(f"ES BM25 加载同义词: {len(self._synonyms)} 个词条")
        except Exception as e:
            logger.warning(f"加载同义词失败: {e}")

    def _init_client(self):
        """初始化 ES 客户端"""
        try:
            if self.es_user and self.es_password:
                self._client = Elasticsearch(
                    [f"http://{self.es_host}:{self.es_port}"],
                    basic_auth=(self.es_user, self.es_password),
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )
            else:
                self._client = Elasticsearch(
                    [f"http://{self.es_host}:{self.es_port}"],
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True
                )

            # 测试连接
            if self._client.ping():
                logger.info(f"Elasticsearch 连接成功: {self.es_host}:{self.es_port}")
            else:
                logger.warning("Elasticsearch 连接失败")
                self._client = None

        except Exception as e:
            logger.error(f"Elasticsearch 初始化失败: {e}")
            self._client = None

    def _get_index_name(self, index_name: str) -> str:
        """获取 ES 索引名称"""
        # 清理索引名称，只保留字母数字和下划线
        safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in index_name)
        return f"{self._index_prefix}{safe_name}"

    def _get_analyzer(self) -> str:
        """获取分词器配置"""
        if self.use_jieba:
            # 使用 ik 或自定义 jieba 插件
            return "ik_max_word"  # 需要安装 elasticsearch-analysis-ik
        else:
            return "standard"

    def create_bm25_index(self, index_name: str, vector_dim: int = None) -> bool:
        """
        创建带 BM25 索引的 Elasticsearch 索引

        Args:
            index_name: 索引名称
            vector_dim: 向量维度（可选，用于混合检索）

        Returns:
            是否创建成功
        """
        es_index = self._get_index_name(index_name)

        if not self._client:
            logger.error("ES 客户端未初始化")
            return False

        # 检查索引是否已存在
        if self._client.indices.exists(index=es_index):
            logger.info(f"ES BM25 索引已存在: {es_index}")
            return True

        # 定义 mapping
        properties = {
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "content": {
                "type": "text",
                "analyzer": "ik_max_word" if self.use_jieba else "standard",
                "search_analyzer": "ik_smart" if self.use_jieba else "standard",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                }
            },
            "content_with_weight": {
                "type": "text",
                "analyzer": "ik_max_word" if self.use_jieba else "standard",
                "search_analyzer": "ik_smart" if self.use_jieba else "standard"
            },
            "document_name": {"type": "keyword"},
            "docnm": {"type": "keyword"},
            "docnm_kwd": {"type": "keyword"},
            "kb_id": {"type": "keyword"},
            "page_num": {"type": "integer"},
            "chunk_index": {"type": "integer"},
            "token_count": {"type": "integer"},
            "created_at": {"type": "date"},
            "source": {"type": "keyword"}
        }

        # 添加向量字段（如果启用混合检索）
        if vector_dim:
            properties["vector"] = {
                "type": "dense_vector",
                "dims": vector_dim,
                "index": True,
                "similarity": "cosine"
            }

        # 创建索引
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index": {
                "max_result_window": 10000
            },
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "ik_max_word" if self.use_jieba else "standard"
                    },
                    "default_search": {
                        "type": "ik_smart" if self.use_jieba else "standard"
                    }
                }
            }
        }

        # 添加同义词过滤器
        if self.use_synonyms and self._synonyms:
            synonym_rules = self._build_synonym_rules()
            if synonym_rules:
                settings["analysis"] = settings.get("analysis", {})
                settings["analysis"]["filter"] = {
                    "synonym_filter": {
                        "type": "synonym",
                        "synonyms": synonym_rules
                    }
                }
                settings["analysis"]["analyzer"]["default_with_synonym"] = {
                    "tokenizer": "ik_max_word" if self.use_jieba else "standard",
                    "filter": ["lowercase", "synonym_filter"]
                }
                properties["content"]["analyzer"] = "default_with_synonym"

        try:
            self._client.indices.create(
                index=es_index,
                mappings={
                    "properties": properties
                },
                settings=settings
            )
            logger.info(f"ES BM25 索引创建成功: {es_index}")
            return True

        except es_exceptions.RequestError as e:
            logger.error(f"创建 ES BM25 索引失败: {e}")
            return False

    def _build_synonym_rules(self) -> List[str]:
        """构建同义词规则"""
        rules = []
        for word, synonyms in self._synonyms.items():
            if synonyms:
                # 格式: word => syn1, syn2
                rule = f"{word} => {', '.join(synonyms)}"
                rules.append(rule)
                # 反向也添加
                for syn in synonyms:
                    if syn != word:
                        reverse_rule = f"{syn} => {word}"
                        rules.append(reverse_rule)
        return rules[:100]  # 限制数量

    def index_documents(
            self,
            documents: List[Dict[str, Any]],
            index_name: str,
            batch_size: int = 100
    ) -> int:
        """
        批量索引文档到 ES

        Args:
            documents: 文档列表
            index_name: 索引名称
            batch_size: 批量大小

        Returns:
            索引的文档数量
        """
        es_index = self._get_index_name(index_name)

        if not self._client:
            logger.error("ES 客户端未初始化")
            return 0

        if not documents:
            return 0

        # 获取向量维度（如果有）
        vector_dim = None
        for doc in documents:
            if "vector" in doc and doc["vector"]:
                vector_dim = len(doc["vector"])
                break

        # 确保索引存在
        self.create_bm25_index(index_name, vector_dim)

        success_count = 0

        # 批量索引
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            try:
                # 使用 bulk API
                from elasticsearch.helpers import bulk

                actions = []
                for doc in batch:
                    # 构建 ES 文档
                    es_doc = {
                        "_index": es_index,
                        "_id": doc.get("id", doc.get("_id", "")),
                        "_source": {
                            "doc_id": doc.get("doc_id", doc.get("id", "")),
                            "chunk_id": doc.get("chunk_id", doc.get("id", "")),
                            "content": doc.get("content", ""),
                            "content_with_weight": doc.get("content_with_weight", doc.get("content", "")),
                            "document_name": doc.get("document_name", doc.get("docnm", "")),
                            "docnm": doc.get("docnm", ""),
                            "docnm_kwd": doc.get("docnm_kwd", ""),
                            "kb_id": doc.get("kb_id", index_name),
                            "page_num": doc.get("page_num", 0),
                            "chunk_index": doc.get("chunk_index", 0),
                            "token_count": doc.get("token_count", 0),
                            "created_at": doc.get("created_at", ""),
                            "source": doc.get("source", "rag")
                        }
                    }

                    # 添加向量（如果有）
                    if "vector" in doc and doc["vector"]:
                        es_doc["_source"]["vector"] = doc["vector"]

                    actions.append(es_doc)

                success, failed = bulk(self._client, actions, stats_only=True, raise_on_error=False)
                success_count += success

                if failed > 0:
                    logger.warning(f"批量索引失败: {failed} 条")

            except Exception as e:
                logger.error(f"批量索引错误: {e}")

        logger.info(f"ES BM25 索引完成: {success_count} 条文档 -> {es_index}")
        return success_count

    def search(
            self,
            query: str,
            index_name: str,
            top_k: int = 10,
            filter_condition: Optional[Dict] = None,
            highlight: bool = True,
            min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        ES BM25 搜索

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回数量
            filter_condition: 过滤条件
            highlight: 是否高亮
            min_score: 最低分数阈值

        Returns:
            搜索结果列表
        """
        es_index = self._get_index_name(index_name)

        if not self._client:
            logger.error("ES 客户端未初始化")
            return []

        if not self._client.indices.exists(index=es_index):
            logger.warning(f"ES BM25 索引不存在: {es_index}")
            return []

        # 查询预处理
        processed_query = self._preprocess_query(query)

        # 构建查询
        must_queries = [
            {
                "multi_match": {
                    "query": processed_query,
                    "fields": ["content^2", "content_with_weight^1.5", "document_name^1"],
                    "type": "best_fields",
                    "operator": "or",
                    "fuzziness": "AUTO"
                }
            }
        ]

        # 添加过滤条件
        filter_queries = []
        if filter_condition:
            for field, value in filter_condition.items():
                if isinstance(value, list):
                    filter_queries.append({"terms": {field: value}})
                else:
                    filter_queries.append({"term": {field: value}})

        # 构建完整查询
        search_body = {
            "query": {
                "bool": {
                    "must": must_queries,
                    "filter": filter_queries if filter_queries else None
                }
            },
            "size": top_k,
            "min_score": min_score
        }

        # 添加高亮
        if highlight:
            search_body["highlight"] = {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 2,
                        "pre_tags": ["<em>"],
                        "post_tags": ["</em>"]
                    }
                }
            }

        try:
            response = self._client.search(index=es_index, body=search_body)

            results = []
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                result = {
                    "_id": hit.get("_id"),
                    "_score": hit.get("_score", 0),
                    "content": source.get("content", ""),
                    "content_with_weight": source.get("content_with_weight", ""),
                    "document_name": source.get("document_name", source.get("docnm", "")),
                    "docnm": source.get("docnm", ""),
                    "docnm_kwd": source.get("docnm_kwd", ""),
                    "chunk_id": source.get("chunk_id", ""),
                    "kb_id": source.get("kb_id", ""),
                    "chunk_index": source.get("chunk_index", 0),
                    "token_count": source.get("token_count", 0),
                    "bm25_score": hit.get("_score", 0),
                    "_search_type": "es_bm25"
                }

                # 添加高亮
                if "highlight" in hit:
                    result["highlights"] = hit["highlight"]

                results.append(result)

            logger.info(f"ES BM25 搜索完成: query={query[:50]}..., 结果数={len(results)}")
            return results

        except Exception as e:
            logger.error(f"ES BM25 搜索失败: {e}")
            return []

    def _preprocess_query(self, query: str) -> str:
        """
        查询预处理：分词、同义词扩展

        Args:
            query: 原始查询

        Returns:
            处理后的查询
        """
        if not query:
            return ""

        processed = query.strip()

        # 1. 使用 jieba 分词提高中文检索效果
        if self.use_jieba:
            words = jieba.lcut_for_search(query)
            # 去重并保留有意义的词
            words = list(set([w for w in words if len(w) > 1]))
            processed = " ".join(words)

        # 2. 同义词扩展（添加到查询中）
        if self.use_synonyms and self._synonyms:
            expanded_terms = []
            original_words = processed.split()

            for word in original_words:
                expanded_terms.append(word)
                # 查找同义词
                for key, synonyms in self._synonyms.items():
                    if word == key or word in key or key in word:
                        expanded_terms.extend(synonyms[:2])  # 最多添加2个同义词
                        break
                    for syn in synonyms:
                        if word == syn or word in syn:
                            expanded_terms.append(key)
                            break

            # 去重
            expanded_terms = list(dict.fromkeys(expanded_terms))
            processed = " ".join(expanded_terms)

        return processed

    def hybrid_search(
            self,
            query: str,
            index_name: str,
            query_vector: List[float] = None,
            top_k: int = 10,
            bm25_weight: float = 0.4,
            vector_weight: float = 0.6,
            filter_condition: Optional[Dict] = None,
            min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        混合检索：ES BM25 + 向量检索

        Args:
            query: 查询文本
            index_name: 索引名称
            query_vector: 查询向量
            top_k: 返回数量
            bm25_weight: BM25 权重
            vector_weight: 向量权重
            filter_condition: 过滤条件
            min_score: 最低分数阈值

        Returns:
            混合检索结果
        """
        es_index = self._get_index_name(index_name)

        if not self._client:
            logger.error("ES 客户端未初始化")
            return []

        if not self._client.indices.exists(index=es_index):
            logger.warning(f"ES BM25 索引不存在: {es_index}")
            return []

        processed_query = self._preprocess_query(query)

        # 构建查询
        should_queries = [
            {
                "multi_match": {
                    "query": processed_query,
                    "fields": ["content^2", "content_with_weight^1.5"],
                    "boost": bm25_weight
                }
            }
        ]

        # 添加向量检索
        if query_vector:
            should_queries.append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    },
                    "boost": vector_weight
                }
            })

        # 构建过滤条件
        filter_queries = []
        if filter_condition:
            for field, value in filter_condition.items():
                if isinstance(value, list):
                    filter_queries.append({"terms": {field: value}})
                else:
                    filter_queries.append({"term": {field: value}})

        search_body = {
            "query": {
                "bool": {
                    "should": should_queries,
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
                result = {
                    "_id": hit.get("_id"),
                    "_score": hit.get("_score", 0),
                    "content": source.get("content", ""),
                    "content_with_weight": source.get("content_with_weight", ""),
                    "document_name": source.get("document_name", source.get("docnm", "")),
                    "chunk_id": source.get("chunk_id", ""),
                    "kb_id": source.get("kb_id", ""),
                    "bm25_score": hit.get("_score", 0) * bm25_weight if bm25_weight > 0 else 0,
                    "_search_type": "es_hybrid"
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"ES 混合检索失败: {e}")
            return []

    def delete_index(self, index_name: str) -> bool:
        """删除 ES BM25 索引"""
        es_index = self._get_index_name(index_name)

        if not self._client:
            return False

        try:
            if self._client.indices.exists(index=es_index):
                self._client.indices.delete(index=es_index)
                logger.info(f"ES BM25 索引已删除: {es_index}")
            return True
        except Exception as e:
            logger.error(f"删除 ES BM25 索引失败: {e}")
            return False

    def delete_documents(self, index_name: str, doc_ids: List[str]) -> int:
        """删除指定文档"""
        es_index = self._get_index_name(index_name)

        if not self._client:
            return 0

        try:
            from elasticsearch.helpers import bulk

            actions = []
            for doc_id in doc_ids:
                actions.append({
                    "_op_type": "delete",
                    "_index": es_index,
                    "_id": doc_id
                })

            success, failed = bulk(self._client, actions, stats_only=True, raise_on_error=False)
            logger.info(f"删除文档: {success} 成功, {failed} 失败")
            return success

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return 0

    def get_document_count(self, index_name: str) -> int:
        """获取文档数量"""
        es_index = self._get_index_name(index_name)

        if not self._client:
            return 0

        try:
            if not self._client.indices.exists(index=es_index):
                return 0

            response = self._client.count(index=es_index)
            return response.get("count", 0)

        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    def is_available(self) -> bool:
        """检查 ES 是否可用"""
        return self._client is not None and self._client.ping()

    def close(self):
        """关闭 ES 连接"""
        if self._client:
            self._client.close()
            logger.info("ES 连接已关闭")


# 全局实例
_es_bm25_retriever = None


def get_es_bm25_retriever() -> ESBM25Retriever:
    """获取 ES BM25 检索器实例"""
    global _es_bm25_retriever
    if _es_bm25_retriever is None:
        _es_bm25_retriever = ESBM25Retriever()
    return _es_bm25_retriever


__all__ = ['ESBM25Retriever', 'get_es_bm25_retriever']