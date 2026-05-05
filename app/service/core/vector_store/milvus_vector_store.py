# app/service/core/vector_store/milvus_vector_store.py
"""Milvus 向量数据库存储实现"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Milvus 向量数据库存储"""

    def __init__(self, host: str = None, port: str = None, user: str = "", password: str = ""):
        """
        初始化 Milvus 连接

        Args:
            host: Milvus 主机地址
            port: Milvus 端口
            user: 用户名
            password: 密码
        """
        self.host = host or os.getenv("VECTOR_STORE_HOST", "localhost")
        self.port = port or os.getenv("VECTOR_STORE_PORT", "19530")
        self.user = user or os.getenv("VECTOR_STORE_USER", "")
        self.password = password or os.getenv("VECTOR_STORE_PASSWORD", "")
        self._connected = False

        self._connect()

    def _connect(self):
        """建立 Milvus 连接"""
        try:
            if self.user and self.password:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password
                )
            else:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port
                )
            self._connected = True
            logger.info(f"Milvus 连接成功: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise

    def _ensure_connected(self):
        """确保连接有效"""
        if not self._connected:
            self._connect()

    def create_index(self, index_name: str, vector_dim: int = None,
                     metric_type: str = "COSINE", **kwargs) -> bool:
        """创建集合 - 确保索引正确创建"""
        self._ensure_connected()

        if vector_dim is None:
            vector_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

        # 检查集合是否存在
        if utility.has_collection(index_name):
            logger.info(f"集合已存在: {index_name}")
            # 关键修复：即使集合存在，也要检查索引是否存在
            collection = Collection(index_name)
            try:
                # 检查索引是否存在
                collection.index()
                logger.info(f"索引已存在，跳过创建")
            except Exception as e:
                logger.warning(f"索引不存在，正在创建: {e}")
                # 创建索引
                index_params = {
                    "metric_type": metric_type,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                collection.create_index("vector", index_params)
                logger.info(f"索引创建成功")
            return True

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="content_with_weight", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="docnm", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="docnm_kwd", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="kb_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="token_count", dtype=DataType.INT32),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="create_timestamp_flt", dtype=DataType.FLOAT),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        ]

        # 创建集合 schema
        schema = CollectionSchema(fields, description=f"RAG 文档集合: {index_name}")
        collection = Collection(index_name, schema)

        # 关键修复：创建索引（必须在插入数据之前，或者插入之后但搜索之前）
        index_params = {
            "metric_type": metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("vector", index_params)

        logger.info(f"集合创建成功: {index_name} (维度:{vector_dim})")
        return True

    def insert(self, documents: List[Dict[str, Any]], index_name: str) -> int:
        """批量插入文档"""
        self._ensure_connected()

        if not documents:
            return 0

        # 确保集合存在
        vector_dim = None
        for doc in documents:
            # 查找向量字段
            if "vector" in doc and isinstance(doc["vector"], list):
                vector_dim = len(doc["vector"])
                break
            for key in doc:
                if key.endswith("_vec") and isinstance(doc[key], list):
                    vector_dim = len(doc[key])
                    # 统一使用 vector 字段
                    doc["vector"] = doc.pop(key)
                    break

        if vector_dim is None:
            logger.error("无法获取向量维度")
            return 0

        self.create_index(index_name, vector_dim)

        import xxhash

        # 准备插入数据 - 每个字段对应一个列表
        ids = []
        contents = []
        contents_weight = []
        docnms = []
        docnm_kwds = []
        doc_ids = []
        kb_ids = []
        token_counts = []
        chunk_indices = []
        timestamps = []
        vectors = []

        for doc in documents:
            # 获取或生成 ID（必须是字符串）
            doc_id = doc.get("id", "")
            if not doc_id:
                content = doc.get("content", "") or doc.get("content_with_weight", "")
                doc_id = xxhash.xxh64(content.encode("utf-8")).hexdigest()

            # 确保 ID 是字符串且不超过长度
            doc_id = str(doc_id)[:200]

            # 获取向量
            vector = doc.get("vector", [])
            if not vector:
                # 如果没有向量，跳过这个文档
                logger.warning(f"文档 {doc_id} 没有向量数据，跳过")
                continue

            # 确保向量是浮点数列表
            vector = [float(v) for v in vector]

            # 添加到各个列表
            ids.append(doc_id)
            contents.append(self._truncate_string(doc.get("content", ""), 65535))
            contents_weight.append(self._truncate_string(doc.get("content_with_weight", ""), 65535))
            docnms.append(self._truncate_string(doc.get("docnm", ""), 500))
            docnm_kwds.append(self._truncate_string(doc.get("docnm_kwd", ""), 500))
            doc_ids.append(self._truncate_string(doc.get("doc_id", ""), 200))
            kb_ids.append(self._truncate_string(doc.get("kb_id", ""), 200))
            token_counts.append(int(doc.get("token_count", 0)))
            chunk_indices.append(int(doc.get("chunk_index", 0)))
            timestamps.append(float(doc.get("create_timestamp_flt", 0.0)))
            vectors.append(vector)

        # 如果没有有效的向量数据，返回 0
        if not ids:
            logger.warning("没有有效的向量数据")
            return 0

        try:
            collection = Collection(index_name)

            # 使用列表格式插入（每个字段是一个列表）
            collection.insert([
                ids,  # id
                contents,  # content
                contents_weight,  # content_with_weight
                docnms,  # docnm
                docnm_kwds,  # docnm_kwd
                doc_ids,  # doc_id
                kb_ids,  # kb_id
                token_counts,  # token_count
                chunk_indices,  # chunk_index
                timestamps,  # create_timestamp_flt
                vectors  # vector
            ])

            collection.flush()
            inserted = len(ids)
            logger.info(f"批量插入成功: {inserted} 条文档 -> {index_name}")
            return inserted
        except Exception as e:
            logger.error(f"Milvus 插入失败: {e}")
            # 打印调试信息
            logger.error(f"数据大小: ids={len(ids)}, vectors={len(vectors)}")
            if ids:
                logger.error(f"ID 示例: {ids[0]}, 类型: {type(ids[0])}")
            if vectors:
                logger.error(f"Vector 示例长度: {len(vectors[0])}")
            return 0

    def _truncate_string(self, s: str, max_length: int) -> str:
        """截断字符串到指定长度"""
        if not s:
            return ""
        s = str(s)
        if len(s) > max_length:
            return s[:max_length - 3] + "..."
        return s

    def delete(self, index_name: str, condition: Dict[str, Any]) -> int:
        """删除符合条件的文档"""
        self._ensure_connected()

        try:
            if not utility.has_collection(index_name):
                return 0

            collection = Collection(index_name)
            collection.load()

            # 构建删除表达式
            expr_parts = []
            for field, value in condition.items():
                if isinstance(value, list):
                    values_str = ", ".join([f"'{v}'" for v in value])
                    expr_parts.append(f"{field} in [{values_str}]")
                else:
                    expr_parts.append(f"{field} == '{value}'")

            expr = " and ".join(expr_parts) if expr_parts else ""

            if expr:
                collection.delete(expr)
                collection.flush()
                logger.info(f"删除文档: 条件 {condition}")
                return 1
            return 0

        except Exception as e:
            logger.error(f"Milvus 删除失败: {e}")
            return 0

    def delete_index(self, index_name: str) -> bool:
        """删除整个集合"""
        self._ensure_connected()

        try:
            if utility.has_collection(index_name):
                utility.drop_collection(index_name)
                logger.info(f"集合删除成功: {index_name}")
            return True
        except Exception as e:
            logger.error(f"集合删除失败: {e}")
            return False

    def search(self, query_vector: List[float], index_name: str, top_k: int = 5,
               filter_condition: Optional[Dict] = None, similarity_threshold: float = 0.5) -> List[Dict]:
        """向量相似度搜索"""
        self._ensure_connected()

        try:
            if not utility.has_collection(index_name):
                logger.warning(f"集合不存在: {index_name}")
                return []

            collection = Collection(index_name)
            collection.load()

            # 确保查询向量是浮点数列表
            query_vector = [float(v) for v in query_vector]

            # 关键修复1：检查向量维度是否匹配
            vector_dim = len(query_vector)
            logger.info(f"搜索: 查询向量维度={vector_dim}, top_k={top_k}, 阈值={similarity_threshold}")

            # 关键修复2：不是归一化，而是使用正确的搜索参数
            # Milvus 的 COSINE 相似度：返回的是 (1 - cosine_distance) 范围 0-2
            # 不需要预先归一化，Milvus 内部会处理

            # 构建搜索参数 - 使用更宽松的参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 20}  # 增加 nprobe 值提高召回率
            }

            # 构建过滤表达式
            expr = self._build_filter_expr(filter_condition) if filter_condition else None

            # 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=[
                    "id", "content", "content_with_weight", "docnm",
                    "docnm_kwd", "doc_id", "kb_id", "token_count",
                    "chunk_index", "create_timestamp_flt"
                ]
            )

            # 格式化结果 - COSINE 返回的分数范围是 0-2，需要转换
            # 转换公式: cosine_similarity = 1 - distance (对于 COSINE metric_type)
            formatted_results = []
            for hits in results:
                logger.info(f"搜索返回 {len(hits)} 个结果")
                for hit in hits:
                    # Milvus 返回的 score 是距离，对于 COSINE 是 1 - cosine_similarity
                    # 所以实际相似度 = 1 - score
                    raw_score = hit.score
                    similarity = 1.0 - raw_score  # 转换回余弦相似度

                    logger.debug(f"  hit: id={hit.id}, raw_score={raw_score:.4f}, similarity={similarity:.4f}")

                    if similarity >= similarity_threshold:
                        doc = {
                            "_id": hit.id,
                            "_score": similarity,  # 使用转换后的相似度
                            "content": hit.entity.get("content", ""),
                            "content_with_weight": hit.entity.get("content_with_weight", ""),
                            "docnm": hit.entity.get("docnm", ""),
                            "docnm_kwd": hit.entity.get("docnm_kwd", ""),
                            "doc_id": hit.entity.get("doc_id", ""),
                            "kb_id": hit.entity.get("kb_id", ""),
                            "token_count": hit.entity.get("token_count", 0),
                            "chunk_index": hit.entity.get("chunk_index", 0),
                            "create_timestamp_flt": hit.entity.get("create_timestamp_flt", 0)
                        }
                        formatted_results.append(doc)

            logger.info(f"向量搜索完成: 召回 {len(formatted_results)} 个文档")
            return formatted_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def _build_filter_expr(self, condition: Dict[str, Any]) -> str:
        """构建过滤表达式"""
        expr_parts = []
        for field, value in condition.items():
            if isinstance(value, list):
                values_str = ", ".join([f"'{v}'" for v in value])
                expr_parts.append(f"{field} in [{values_str}]")
            else:
                expr_parts.append(f"{field} == '{value}'")
        return " and ".join(expr_parts)

    def index_exists(self, index_name: str) -> bool:
        """检查集合是否存在"""
        self._ensure_connected()
        try:
            return utility.has_collection(index_name)
        except Exception as e:
            logger.error(f"检查集合存在失败: {e}")
            return False

    def get_document_count(self, index_name: str) -> int:
        """获取集合中的文档数量"""
        self._ensure_connected()
        try:
            if not utility.has_collection(index_name):
                return 0
            collection = Collection(index_name)
            collection.flush()
            return collection.num_entities
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    def close(self):
        """关闭连接"""
        if self._connected:
            try:
                connections.disconnect("default")
                self._connected = False
                logger.info("Milvus 连接已关闭")
            except Exception as e:
                logger.error(f"关闭 Milvus 连接失败: {e}")


__all__ = ['MilvusVectorStore']