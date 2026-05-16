"""
文档分块器 - 父子分块策略（精简版）
"""

import hashlib
import logging
import re
from typing import List, Dict, Any, Optional

from .types import ParentChildDocument, ParentChunk, ChildChunk

logger = logging.getLogger(__name__)


class ParentChildSplitter:
    """父子分块器"""

    def __init__(
            self,
            parent_chunk_size: int = 500,
            child_chunk_size: int = 150,
            parent_overlap: int = 50,
            child_overlap: int = 20,
            min_child_size: int = 30,
            separators: List[str] = None
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.min_child_size = min_child_size
        self.separators = separators or [
            "\n\n", "\n", "。", "！", "？", "；",
            ". ", "! ", "? ", "; ", "，", ", ", " ",
        ]

    def _count_tokens(self, text: str) -> int:
        """估算token数量"""
        if not text:
            return 0
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

        return self._split_by_length(text, max_size)

    def _split_by_length(self, text: str, max_size: int) -> List[str]:
        """按长度切分"""
        chunks = []
        current = ""
        for ch in text:
            if self._count_tokens(current + ch) <= max_size:
                current += ch
            else:
                if current:
                    chunks.append(current)
                current = ch
        if current:
            chunks.append(current)
        return chunks

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """应用重叠"""
        if overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                prev_tokens = list(prev_chunk)
                overlap_text = ''.join(prev_tokens[-min(overlap * 4, len(prev_tokens)):])
                chunk = overlap_text + chunk
            overlapped.append(chunk)
        return overlapped

    def split_to_parents(self, text: str) -> List[str]:
        """切分为父块"""
        if not text:
            return []
        chunks = self._split_by_separator(text, self.parent_chunk_size)
        return self._apply_overlap(chunks, self.parent_overlap)

    def split_parent_to_children(self, parent_text: str, parent_id: str) -> List[ChildChunk]:
        """将父块切分为子块"""
        if not parent_text:
            return []
        child_texts = self._split_by_separator(parent_text, self.child_chunk_size)
        child_texts = self._apply_overlap(child_texts, self.child_overlap)
        child_texts = [t for t in child_texts if self._count_tokens(t) >= self.min_child_size]

        children = []
        for i, child_text in enumerate(child_texts):
            child_hash = hashlib.md5(f"{parent_id}_{i}_{child_text[:100]}".encode()).hexdigest()[:16]
            children.append(ChildChunk(
                id=f"child_{child_hash}",
                content=child_text,
                parent_id=parent_id,
                chunk_index=i,
                token_count=self._count_tokens(child_text)
            ))
        return children

    def split_document(
            self,
            text: str,
            metadata: Optional[Dict[str, Any]] = None,
            document_id: str = None,
            document_name: str = None
    ) -> ParentChildDocument:
        """将文档切分为父子块结构"""
        if not text:
            return ParentChildDocument(document_id=document_id or "", document_name=document_name or "")

        parent_texts = self.split_to_parents(text)

        parent_chunks = []
        all_children = []

        base_metadata = metadata or {}
        if document_name:
            base_metadata['source'] = document_name
            base_metadata['document_name'] = document_name

        for i, parent_text in enumerate(parent_texts):
            if not parent_text.strip():
                continue

            parent_hash = hashlib.md5(f"{document_id}_{i}_{parent_text[:100]}".encode()).hexdigest()[:16]
            parent_id = f"parent_{parent_hash}"

            children = self.split_parent_to_children(parent_text, parent_id)

            for child in children:
                child.metadata = base_metadata.copy()
                child.metadata['parent_chunk_index'] = i
                child.metadata['child_chunk_index'] = child.chunk_index

            parent_chunks.append(ParentChunk(
                id=parent_id,
                content=parent_text,
                child_ids=[c.id for c in children],
                children=children,
                chunk_index=i,
                token_count=self._count_tokens(parent_text),
                metadata=base_metadata.copy()
            ))
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
            return {"parent_count": 0, "child_count": 0, "avg_parent_tokens": 0, "avg_child_tokens": 0}
        parent_tokens = [p.token_count for p in doc.parent_chunks]
        child_tokens = [c.token_count for c in doc.all_children]
        return {
            "parent_count": len(doc.parent_chunks),
            "child_count": len(doc.all_children),
            "avg_parent_tokens": sum(parent_tokens) / len(parent_tokens) if parent_tokens else 0,
            "avg_child_tokens": sum(child_tokens) / len(child_tokens) if child_tokens else 0,
        }


# ========== 便捷函数 ==========

def chunk_document_parent_child(
        text: str,
        parent_chunk_size: int = 500,
        child_chunk_size: int = 150,
        metadata: Optional[Dict] = None,
        document_id: str = None,
        document_name: str = None
) -> ParentChildDocument:
    """快速父子分块（兼容原函数名）"""
    splitter = ParentChildSplitter(
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size
    )
    return splitter.split_document(text, metadata, document_id, document_name)


def chunk_text_simple(text: str, chunk_size: int = 256) -> List[str]:
    """简单分块（兼容原函数）"""
    splitter = ParentChildSplitter(parent_chunk_size=chunk_size, child_chunk_size=chunk_size)
    return splitter.split_to_parents(text)


__all__ = [
    'ParentChildSplitter',
    'chunk_document_parent_child',
    'chunk_text_simple',
]