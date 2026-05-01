"""
向量处理器
将分块后的文本转换为向量并存储
"""

import xxhash
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .embedding_service import get_embedding_service
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorizedChunk:
    """向量化的文本块"""
    id: str  # 块ID
    content: str  # 原始文本内容
    vector: List[float]  # 向量
    metadata: Dict[str, Any]  # 元数据
    created_at: str  # 创建时间


class VectorProcessor:
    """
    向量处理器
    负责将文本块转换为向量，并准备存储格式
    """

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service or get_embedding_service()

    def process_single_chunk(
            self,
            chunk_id: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[VectorizedChunk]:
        """
        处理单个文本块，生成向量

        Args:
            chunk_id: 块ID
            content: 文本内容
            metadata: 元数据

        Returns:
            VectorizedChunk 对象，失败返回 None
        """
        vector = self.embedding_service.generate_embedding(content)

        if vector is None:
            logger.warning(f"向量生成失败: chunk_id={chunk_id}")
            return None

        return VectorizedChunk(
            id=chunk_id,
            content=content,
            vector=vector,
            metadata=metadata or {},
            created_at=datetime.datetime.now().isoformat()
        )

    def process_chunks_batch(
            self,
            chunks: List[Dict[str, Any]],
            batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        批量处理文本块，生成向量

        Args:
            chunks: 文本块列表，每个块包含 id, content_with_weight, metadata 等字段
            batch_size: 批处理大小

        Returns:
            处理后的块列表（包含向量字段）
        """
        if not chunks:
            return []

        # 提取所有文本内容
        texts = [chunk.get("content_with_weight", "") for chunk in chunks]

        # 批量生成向量
        vectors = self.embedding_service.generate_embeddings(texts)

        # 处理每个块
        processed_chunks = []
        vector_field_name = self.embedding_service.get_vector_field_name()

        for chunk, vector in zip(chunks, vectors):
            if vector is None:
                logger.warning(f"向量生成失败，跳过: {chunk.get('id', 'unknown')}")
                continue

            # 构建 ES 文档格式
            doc = self._build_es_document(chunk, vector, vector_field_name)
            processed_chunks.append(doc)

        logger.info(f"批量向量化完成: {len(processed_chunks)}/{len(chunks)} 成功")
        return processed_chunks

    def _build_es_document(
            self,
            chunk: Dict[str, Any],
            vector: List[float],
            vector_field_name: str
    ) -> Dict[str, Any]:
        """
        构建 Elasticsearch 文档格式

        Args:
            chunk: 原始块数据
            vector: 向量
            vector_field_name: 向量字段名

        Returns:
            ES 文档字典
        """
        # 生成块ID
        chunk_id = chunk.get("id")
        if not chunk_id:
            content = chunk.get("content_with_weight", "")
            kb_id = chunk.get("kb_id", "")
            chunk_id = xxhash.xxh64((content + kb_id).encode("utf-8")).hexdigest()

        doc = {
            "id": chunk_id,
            "content_with_weight": chunk.get("content_with_weight", ""),
            "content_ltks": chunk.get("content_ltks", ""),
            "content_sm_ltks": chunk.get("content_sm_ltks", ""),
            "important_kwd": chunk.get("important_kwd", []),
            "important_tks": chunk.get("important_tks", []),
            "question_kwd": chunk.get("question_kwd", []),
            "question_tks": chunk.get("question_tks", []),
            "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "create_timestamp_flt": datetime.datetime.now().timestamp(),
            "kb_id": chunk.get("kb_id", ""),
            "docnm_kwd": chunk.get("docnm_kwd", ""),
            "title_tks": chunk.get("title_tks", ""),
            "doc_id": chunk.get("doc_id", ""),
            "docnm": chunk.get("docnm", ""),
        }

        # 添加向量字段
        doc[vector_field_name] = vector

        # 合并额外元数据
        metadata = chunk.get("metadata", {})
        doc.update(metadata)

        return doc

    def process_rag_chunks(
            self,
            chunks: List[Dict[str, Any]],
            index_name: str,
            file_name: str
    ) -> List[Dict[str, Any]]:
        """
        处理 RAG 项目的块格式（兼容原有 process_items 逻辑）

        Args:
            chunks: 从 chunk() 函数返回的块列表
            index_name: 索引名称
            file_name: 文件名

        Returns:
            处理后的文档列表（包含向量）
        """
        texts = [chunk.get("content_with_weight", "") for chunk in chunks]
        vectors = self.embedding_service.generate_embeddings(texts)

        processed = []
        vector_field_name = self.embedding_service.get_vector_field_name()

        for chunk, vector in zip(chunks, vectors):
            if vector is None:
                continue

            # 生成 chunk_id
            content = chunk.get("content_with_weight", "")
            chunk_id = xxhash.xxh64((content + index_name).encode("utf-8")).hexdigest()

            doc = {
                "id": chunk_id,
                "content_ltks": chunk.get("content_ltks", ""),
                "content_with_weight": content,
                "content_sm_ltks": chunk.get("content_sm_ltks", ""),
                "important_kwd": [],
                "important_tks": [],
                "question_kwd": [],
                "question_tks": [],
                "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "create_timestamp_flt": datetime.datetime.now().timestamp(),
                "kb_id": index_name,
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "title_tks": chunk.get("title_tks", ""),
                "doc_id": xxhash.xxh64(file_name.encode("utf-8")).hexdigest(),
                "docnm": file_name,
            }

            # 添加向量
            doc[vector_field_name] = vector

            processed.append(doc)

        logger.info(f"处理完成: {len(processed)} 个文档块已向量化")
        return processed