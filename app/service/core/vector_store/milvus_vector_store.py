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
        """创建集合 - 修复索引问题"""
        self._ensure_connected()

        if vector_dim is None:
            vector_dim = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

        # 检查集合是否存在
        if utility.has_collection(index_name):
            logger.info(f"集合已存在: {index_name}")
            collection = Collection(index_name)
            try:
                # 确保索引存在
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
                collection.load()  # 重要：加载集合到内存
                logger.info(f"索引创建成功并已加载")
            return True

        # 定义字段
        fields = [
            FieldSchema(name="user_level", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="content_with_weight", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="parent_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="parent_chunk_index", dtype=DataType.INT32),
            FieldSchema(name="child_chunk_index", dtype=DataType.INT32),
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

        # 创建索引
        index_params = {
            "metric_type": metric_type,
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("vector", index_params)

        # 重要：加载集合到内存，使其立即可搜索
        collection.load()

        logger.info(f"集合创建成功: {index_name} (维度:{vector_dim})")
        return True

    def insert(self, documents: List[Dict[str, Any]], index_name: str, user_level: str = "normal") -> int:
        """批量插入文档 - 确保插入后集合可搜索"""
        self._ensure_connected()

        if not documents:
            return 0

        # 为每个文档添加 user_level
        for doc in documents:
            if "user_level" not in doc:
                doc["user_level"] = user_level

        # 确保集合存在
        vector_dim = None
        for doc in documents:
            if "vector" in doc and isinstance(doc["vector"], list):
                vector_dim = len(doc["vector"])
                break

        if vector_dim is None:
            logger.error("无法获取向量维度")
            return 0

        # 创建或获取集合
        self.create_index(index_name, vector_dim)

        import xxhash

        # 准备插入数据
        user_levels = []
        ids = []
        contents = []
        contents_weight = []
        parent_ids = []
        parent_contents = []
        chunk_types = []
        parent_chunk_indices = []
        child_chunk_indices = []
        docnms = []
        docnm_kwds = []
        doc_ids = []
        kb_ids = []
        token_counts = []
        chunk_indices = []
        timestamps = []
        vectors = []

        for doc in documents:
            doc_id = doc.get("id", "")
            if not doc_id:
                content = doc.get("content", "") or doc.get("content_with_weight", "")
                doc_id = xxhash.xxh64(content.encode("utf-8")).hexdigest()

            doc_id = str(doc_id)[:200]

            vector = doc.get("vector", [])
            if not vector:
                logger.warning(f"文档 {doc_id} 没有向量数据，跳过")
                continue

            vector = [float(v) for v in vector]

            # 获取 user_level
            level = doc.get("user_level", user_level)
            user_levels.append(self._truncate_string(level, 20))

            ids.append(doc_id)
            contents.append(self._truncate_string(doc.get("content", ""), 65535))
            contents_weight.append(self._truncate_string(doc.get("content_with_weight", ""), 65535))

            # 父子分块字段
            parent_ids.append(self._truncate_string(doc.get("parent_id", ""), 200))
            parent_contents.append(self._truncate_string(doc.get("parent_content", ""), 65535))
            chunk_types.append(self._truncate_string(doc.get("chunk_type", "child"), 20))
            parent_chunk_indices.append(int(doc.get("parent_chunk_index", 0)))
            child_chunk_indices.append(int(doc.get("child_chunk_index", 0)))

            docnms.append(self._truncate_string(doc.get("docnm", ""), 500))
            docnm_kwds.append(self._truncate_string(doc.get("docnm_kwd", ""), 500))
            doc_ids.append(self._truncate_string(doc.get("doc_id", ""), 200))
            kb_ids.append(self._truncate_string(doc.get("kb_id", index_name), 200))
            token_counts.append(int(doc.get("token_count", 0)))
            chunk_indices.append(int(doc.get("chunk_index", 0)))
            timestamps.append(float(doc.get("create_timestamp_flt", 0.0)))
            vectors.append(vector)

        if not ids:
            logger.warning("没有有效的向量数据")
            return 0

        try:
            collection = Collection(index_name)

            # 插入数据（包含父子分块字段）
            collection.insert([
                user_levels,
                ids,
                contents,
                contents_weight,
                parent_ids,
                parent_contents,
                chunk_types,
                parent_chunk_indices,
                child_chunk_indices,
                docnms,
                docnm_kwds,
                doc_ids,
                kb_ids,
                token_counts,
                chunk_indices,
                timestamps,
                vectors
            ])

            collection.flush()

            # 重要：插入后重新加载集合
            collection.load()

            inserted = len(ids)
            logger.info(f"批量插入成功: {inserted} 条文档 -> {index_name}")
            return inserted

        except Exception as e:
            logger.error(f"Milvus 插入失败: {e}")
            import traceback
            traceback.print_exc()
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

    # app/service/core/vector_store/milvus_vector_store.py

    def batch_delete_by_docnm(self, index_name: str, filenames: List[str]) -> Dict[str, int]:
        """
        批量删除多个文档的所有分块

        Args:
            index_name: 索引名称
            filenames: 文件名列表

        Returns:
            Dict[str, int]: 每个文件删除的记录数
        """
        self._ensure_connected()

        if not filenames:
            return {}

        try:
            if not utility.has_collection(index_name):
                logger.warning(f"集合不存在: {index_name}")
                return {filename: 0 for filename in filenames}

            collection = Collection(index_name)
            collection.load()

            # 构建 OR 条件: docnm == 'file1' or docnm == 'file2' or docnm == 'file3'
            expr_parts = [f"docnm == '{filename}'" for filename in filenames]
            expr = " or ".join(expr_parts)

            logger.info(f"批量删除文档: 条件 {expr}")

            # 先查询每个文件有多少条记录
            result_counts = {}
            for filename in filenames:
                count_expr = f"docnm == '{filename}'"
                result = collection.query(
                    expr=count_expr,
                    output_fields=["docnm"],
                    limit=10000
                )
                result_counts[filename] = len(result)

            # 执行批量删除
            collection.delete(expr)
            collection.flush()

            logger.info(f"批量删除完成: {result_counts}")
            return result_counts

        except Exception as e:
            logger.error(f"Milvus 批量删除失败: {e}")
            return {filename: 0 for filename in filenames}

    def search(self, query_vector: List[float], index_name: str, top_k: int = 5,
               filter_condition: Optional[Dict] = None, similarity_threshold: float = 0.5,
               user_level: str = None) -> List[Dict]:
        """向量相似度搜索（支持用户等级过滤）"""
        self._ensure_connected()

        try:
            if not utility.has_collection(index_name):
                logger.warning(f"集合不存在: {index_name}")
                return []

            collection = Collection(index_name)
            collection.load()

            # 确保查询向量是浮点数列表
            query_vector = [float(v) for v in query_vector]

            # 构建过滤表达式
            expr_parts = []

            # 添加用户等级过滤
            if user_level:
                level_priority = {"normal": 1, "admin": 2, "owner": 3}
                current_priority = level_priority.get(user_level, 1)

                # 允许访问等级 <= 当前用户等级的文档
                allowed_levels = [level for level, priority in level_priority.items() if priority <= current_priority]
                levels_str = ", ".join([f"'{level}'" for level in allowed_levels])
                expr_parts.append(f"user_level in [{levels_str}]")

            # 添加其他过滤条件
            if filter_condition:
                for field, value in filter_condition.items():
                    if isinstance(value, list):
                        values_str = ", ".join([f"'{v}'" for v in value])
                        expr_parts.append(f"{field} in [{values_str}]")
                    else:
                        expr_parts.append(f"{field} == '{value}'")

            expr = " and ".join(expr_parts) if expr_parts else None

            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 20}
            }

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
                    "chunk_index", "create_timestamp_flt", "user_level",
                    "parent_id","parent_content"
                ]
            )

            formatted_results = []
            for hits in results:
                for hit in hits:
                    raw_score = hit.score
                    similarity = 1.0 - raw_score

                    if similarity >= similarity_threshold:
                        doc = {
                            "_id": hit.id,
                            "_score": similarity,
                            "content": hit.entity.get("content", ""),
                            "content_with_weight": hit.entity.get("content_with_weight", ""),
                            "docnm": hit.entity.get("docnm", ""),
                            "docnm_kwd": hit.entity.get("docnm_kwd", ""),
                            "doc_id": hit.entity.get("doc_id", ""),
                            "kb_id": hit.entity.get("kb_id", ""),
                            "token_count": hit.entity.get("token_count", 0),
                            "chunk_index": hit.entity.get("chunk_index", 0),
                            "create_timestamp_flt": hit.entity.get("create_timestamp_flt", 0),
                            "user_level": hit.entity.get("user_level", "normal"),
                            # ========== 关键修复：添加 parent_id 和 parent_content ==========
                            "parent_id": hit.entity.get("parent_id", ""),
                            "parent_content": hit.entity.get("parent_content", "")
                        }
                        logger.info(
                            f"检索结果: id={hit.id}, docnm={doc['docnm']}, parent_id={doc['parent_id'][:20] if doc['parent_id'] else 'N/A'}, similarity={similarity}")
                        formatted_results.append(doc)

            logger.info(f"向量搜索完成: 召回 {len(formatted_results)} 个文档 (user_level_filter={user_level})")
            return formatted_results

        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            import traceback
            traceback.print_exc()
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