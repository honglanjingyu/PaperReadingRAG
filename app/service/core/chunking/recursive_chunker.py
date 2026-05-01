# app/service/core/chunking/recursive_chunker.py
"""
递归分块器 - 在自然边界处切分
"""

import re
import hashlib
from typing import List, Dict, Any, Optional

from .chunk_strategies import BaseChunker, Chunk


class RecursiveChunkerSimple:
    """简单的递归分块器 - 用于独立使用"""

    def __init__(self, chunk_token_num: int = 256):
        self.chunk_token_num = chunk_token_num
        self.min_chunk_size = 20
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ", ", " "]

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        return self._recursive_split(text)

    def _recursive_split(self, text: str) -> List[str]:
        if self._count_tokens(text) <= self.chunk_token_num:
            return [text]

        for separator in self.separators:
            if separator in text:
                parts = text.split(separator, 1)
                left, right = parts[0], parts[1]
                if self._count_tokens(left) >= self.min_chunk_size:
                    return self._recursive_split(left) + self._recursive_split(right)

        mid = len(text) // 2
        return self._recursive_split(text[:mid]) + self._recursive_split(text[mid:])

    def chunk_to_vector_chunks(self, text: str, metadata: Dict = None) -> List:
        """分块并返回 VectorChunk 对象列表"""
        from app.service.core.embedding.vector_types import VectorChunk

        chunk_texts = self.chunk(text)
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            if chunk_text.strip():
                chunk_id = hashlib.md5(f"{i}_{chunk_text[:100]}".encode()).hexdigest()[:16]
                chunks.append(VectorChunk(
                    id=f"chunk_{i}_{chunk_id}",
                    content=chunk_text,
                    metadata=metadata or {},
                    token_count=self._count_tokens(chunk_text),
                    chunk_index=i
                ))
        return chunks


__all__ = ['RecursiveChunkerSimple']