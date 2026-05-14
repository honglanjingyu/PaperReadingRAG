# app/service/core/chunking/parent_child_chunker.py
"""
父子分块器 - 支持父子文档结构
核心思想：
- 子块：短小精悍，用于精准检索
- 父块：内容完整，用于提供充足上下文
"""

import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import logging

logger = logging.getLogger(__name__)


@dataclass
class ChildChunk:
    """子块 - 用于检索"""
    id: str
    content: str
    parent_id: str
    chunk_index: int
    start_pos: int
    end_pos: int
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentChunk:
    """父块 - 用于生成答案"""
    id: str
    content: str
    child_ids: List[str] = field(default_factory=list)
    children: List[ChildChunk] = field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentChildDocument:
    """父子文档结构"""
    document_id: str
    document_name: str
    parent_chunks: List[ParentChunk] = field(default_factory=list)
    all_children: List[ChildChunk] = field(default_factory=list)

    def get_parent_by_child_id(self, child_id: str) -> Optional[ParentChunk]:
        """根据子块ID获取父块"""
        for parent in self.parent_chunks:
            if child_id in parent.child_ids:
                return parent
        return None


class ParentChildChunker:
    """
    父子分块器

    分块策略：
    1. 先将文档切分为父块（较大粒度，如500-800 tokens）
    2. 再将每个父块切分为子块（较小粒度，如150-250 tokens）
    3. 保持父子关系映射
    """

    def __init__(
            self,
            parent_chunk_size: int = 500,  # 父块大小（tokens）
            child_chunk_size: int = 150,  # 子块大小（tokens）
            parent_overlap: int = 50,  # 父块重叠
            child_overlap: int = 20,  # 子块重叠
            min_child_size: int = 30,  # 最小子块大小
            separators: List[str] = None
    ):
        """
        初始化父子分块器

        Args:
            parent_chunk_size: 父块目标大小（tokens）
            child_chunk_size: 子块目标大小（tokens）
            parent_overlap: 父块之间的重叠大小
            child_overlap: 子块之间的重叠大小
            min_child_size: 最小子块大小
            separators: 分隔符列表
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.min_child_size = min_child_size

        self.separators = separators or [
            "\n\n",  # 段落
            "\n",  # 换行
            "。", "！", "？", "；",  # 中文句子分隔符
            ". ", "! ", "? ", "; ",  # 英文句子分隔符
            "，", ", ",  # 逗号
            " ",  # 空格
        ]

    def _count_tokens(self, text: str) -> int:
        """估算token数量"""
        if not text:
            return 0
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def _split_by_separator(self, text: str, max_size: int) -> List[str]:
        """按分隔符分割文本"""
        if self._count_tokens(text) <= max_size:
            return [text]

        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                chunks = []
                current = ""

                for part in parts:
                    if not part.strip():
                        continue

                    test_current = current + separator + part if current else part
                    if self._count_tokens(test_current) <= max_size:
                        current = test_current
                    else:
                        if current:
                            chunks.append(current)
                        current = part

                if current:
                    chunks.append(current)

                if len(chunks) > 1:
                    return chunks

        # 没有找到合适的分隔符，按长度切分
        return self._split_by_length(text, max_size)

    def _split_by_length(self, text: str, max_size: int) -> List[str]:
        """按长度切分"""
        chunks = []
        words = list(text)
        current = ""

        for word in words:
            if self._count_tokens(current + word) <= max_size:
                current += word
            else:
                if current:
                    chunks.append(current)
                current = word

        if current:
            chunks.append(current)

        return chunks

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """应用重叠"""
        if overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0 and i < len(chunks):
                # 从上一个chunk末尾取overlap个token
                prev_chunk = chunks[i - 1]
                prev_tokens = list(prev_chunk)
                overlap_text = ''.join(prev_tokens[-min(overlap * 4, len(prev_tokens)):])
                chunk = overlap_text + chunk

            overlapped.append(chunk)

        return overlapped

    def chunk_to_parents(self, text: str) -> List[str]:
        """将文本切分为父块"""
        if not text:
            return []

        # 按分隔符切分
        parent_chunks = self._split_by_separator(text, self.parent_chunk_size)

        # 应用重叠
        parent_chunks = self._apply_overlap(parent_chunks, self.parent_overlap)

        return parent_chunks

    def chunk_parent_to_children(self, parent_text: str, parent_id: str) -> List[ChildChunk]:
        """将父块切分为子块"""
        if not parent_text:
            return []

        # 按分隔符切分子块
        child_texts = self._split_by_separator(parent_text, self.child_chunk_size)

        # 应用重叠
        child_texts = self._apply_overlap(child_texts, self.child_overlap)

        # 过滤太小的子块
        child_texts = [t for t in child_texts if self._count_tokens(t) >= self.min_child_size]

        children = []
        for i, child_text in enumerate(child_texts):
            child_hash = hashlib.md5(f"{parent_id}_{i}_{child_text[:100]}".encode()).hexdigest()[:16]
            children.append(ChildChunk(
                id=f"child_{child_hash}",
                content=child_text,
                parent_id=parent_id,  # 直接使用传入的 parent_id
                chunk_index=i,
                start_pos=0,
                end_pos=len(child_text),
                token_count=self._count_tokens(child_text)
            ))

        return children

    def chunk_document(
            self,
            text: str,
            metadata: Optional[Dict[str, Any]] = None,
            document_id: str = None,
            document_name: str = None
    ) -> ParentChildDocument:
        """
        将文档切分为父子块结构

        Args:
            text: 文档文本内容
            metadata: 元数据（包含文档名等信息）
            document_id: 文档ID
            document_name: 文档名称

        Returns:
            ParentChildDocument 对象
        """
        if not text:
            return ParentChildDocument(
                document_id=document_id or "",
                document_name=document_name or ""
            )

        # 1. 切分为父块
        parent_texts = self.chunk_to_parents(text)

        # 2. 为每个父块创建子块
        parent_chunks = []
        all_children = []

        # 准备基础 metadata（包含文档名）
        base_metadata = metadata or {}
        if document_name:
            base_metadata['source'] = document_name
            base_metadata['document_name'] = document_name

        for i, parent_text in enumerate(parent_texts):
            if not parent_text.strip():
                continue

            # 生成父块ID
            parent_hash = hashlib.md5(f"{document_id}_{i}_{parent_text[:100]}".encode()).hexdigest()[:16]
            parent_id = f"parent_{parent_hash}"

            # 创建子块
            children = self.chunk_parent_to_children(parent_text, parent_id)

            # 为每个子块添加 metadata
            for child in children:
                child.metadata = base_metadata.copy()
                # 添加子块特有的元数据
                child.metadata['parent_chunk_index'] = i
                child.metadata['child_chunk_index'] = child.chunk_index

            parent_chunk = ParentChunk(
                id=parent_id,
                content=parent_text,
                child_ids=[c.id for c in children],
                children=children,
                chunk_index=i,
                token_count=self._count_tokens(parent_text),
                metadata=base_metadata.copy()
            )

            parent_chunks.append(parent_chunk)
            all_children.extend(children)

        logger.info(f"父子分块完成: {len(parent_chunks)} 个父块, {len(all_children)} 个子块")

        return ParentChildDocument(
            document_id=document_id or "",
            document_name=document_name or "",
            parent_chunks=parent_chunks,
            all_children=all_children
        )

    def get_statistics(self, doc: ParentChildDocument) -> Dict[str, Any]:
        """获取分块统计信息"""
        if not doc.parent_chunks:
            return {
                "parent_count": 0,
                "child_count": 0,
                "avg_parent_tokens": 0,
                "avg_child_tokens": 0
            }

        parent_tokens = [p.token_count for p in doc.parent_chunks]
        child_tokens = [c.token_count for c in doc.all_children]

        return {
            "parent_count": len(doc.parent_chunks),
            "child_count": len(doc.all_children),
            "avg_parent_tokens": sum(parent_tokens) / len(parent_tokens) if parent_tokens else 0,
            "avg_child_tokens": sum(child_tokens) / len(child_tokens) if child_tokens else 0,
            "min_parent_tokens": min(parent_tokens) if parent_tokens else 0,
            "max_parent_tokens": max(parent_tokens) if parent_tokens else 0,
            "min_child_tokens": min(child_tokens) if child_tokens else 0,
            "max_child_tokens": max(child_tokens) if child_tokens else 0
        }


def chunk_document_parent_child(
        text: str,
        parent_chunk_size: int = 500,
        child_chunk_size: int = 150,
        metadata: Optional[Dict] = None,
        document_id: str = None,
        document_name: str = None
) -> ParentChildDocument:
    """快速父子分块"""
    chunker = ParentChildChunker(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size
    )
    return chunker.chunk_document(text, metadata, document_id, document_name)


__all__ = [
    'ParentChildChunker',
    'ParentChildDocument',
    'ParentChunk',
    'ChildChunk',
    'chunk_document_parent_child'
]