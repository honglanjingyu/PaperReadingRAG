# app/service/core/chunking/chunk_strategies.py

"""分块策略实现
支持多种分块方式：固定长度、语义、递归、句子、段落
"""

import re
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum

from .chunk_types import Chunk


class ChunkStrategy(Enum):
    """分块策略枚举"""
    FIXED_TOKEN = "fixed_token"  # 固定 token 数
    SEMANTIC = "semantic"  # 语义分块
    RECURSIVE = "recursive"  # 递归分块
    SENTENCE = "sentence"  # 句子级分块
    PARAGRAPH = "paragraph"  # 段落级分块


class BaseChunker:
    """分块器基类"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'chunk_token_num': 128,  # 默认块大小（token数）
            'min_chunk_size': 20,  # 最小块大小
            'overlap_tokens': 20,  # 重叠 token 数
            'delimiter': "\n!?。；！？",  # 分隔符
            **(config or {})
        }

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """分块主方法，子类必须实现"""
        raise NotImplementedError

    def _count_tokens(self, text: str) -> int:
        """估算 token 数量（简化版）"""
        if not text:
            return 0
        # 中文约 1.5 字符/token，英文约 4 字符/token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def _create_chunk(self, text: str, start: int, end: int,
                      metadata: Dict, chunk_id: int) -> Chunk:
        """创建 Chunk 对象"""
        return Chunk(
            id=f"chunk_{chunk_id}",
            content=text[start:end].strip(),
            metadata=metadata or {},
            start_idx=start,
            end_idx=end,
            token_count=self._count_tokens(text[start:end])
        )


class FixedTokenChunker(BaseChunker):
    """固定 Token 数分块策略"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.chunk_size = self.config.get('chunk_token_num', 128)
        self.overlap = self.config.get('overlap_tokens', 20)

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        chunks = []
        start = 0
        length = len(text)
        chunk_id = 0

        while start < length:
            end = min(start + self.chunk_size * 4, length)

            chunk_text = text[start:end]
            chunk_obj = self._create_chunk(chunk_text, start, end,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

            start = end - self.overlap * 4
            chunk_id += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """
    递归分块策略
    优先在段落、句子、短语等自然边界处切分
    """

    def __init__(self, config: Optional[Union[Dict, int]] = None):
        """
        初始化递归分块器

        Args:
            config: 配置字典，或者直接传入 chunk_token_num (整数)
        """
        # 支持直接传入整数作为 chunk_token_num
        if isinstance(config, int):
            config = {'chunk_token_num': config}
        elif config is None:
            config = {}
        super().__init__(config)

        self.separators = [
            "\n\n",  # 段落
            "\n",  # 换行
            "。", "！", "？", "；",  # 中文句子分隔符
            ". ", "! ", "? ", "; ",  # 英文句子分隔符
            "，", ", ",  # 逗号
            " ",  # 空格
        ]

    # ========== 标准接口 ==========

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """分块并返回 Chunk 对象列表"""
        if not text:
            return []

        chunks = []
        current_chunk = ""
        chunk_id = 0

        for separator in self.separators:
            if len(current_chunk) < self.config['chunk_token_num'] * 4:
                parts = text.split(separator)
                for part in parts:
                    if self._count_tokens(current_chunk + part) > self.config['chunk_token_num']:
                        if current_chunk:
                            chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                                           metadata or {}, chunk_id)
                            chunks.append(chunk_obj)
                            current_chunk = part
                            chunk_id += 1
                        else:
                            current_chunk = part
                    else:
                        current_chunk += separator + part if current_chunk else part
            else:
                if current_chunk:
                    chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                                   metadata or {}, chunk_id)
                    chunks.append(chunk_obj)
                    chunk_id += 1
                if parts:
                    remaining = separator.join(parts)
                    sub_chunks = self.chunk(remaining, metadata)
                    for sc in sub_chunks:
                        sc.id = f"chunk_{chunk_id}"
                        chunks.append(sc)
                        chunk_id += 1
                return chunks

        if current_chunk:
            chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

        return chunks

    # ========== 简化接口 ==========

    def chunk_to_texts(self, text: str) -> List[str]:
        """
        分块并返回文本列表（仅返回文本，不创建 Chunk 对象）

        Args:
            text: 输入文本

        Returns:
            分块后的文本列表
        """
        if not text:
            return []
        return self._recursive_split_to_texts(text)

    def _recursive_split_to_texts(self, text: str) -> List[str]:
        """递归分割文本，返回文本列表"""
        if self._count_tokens(text) <= self.config['chunk_token_num']:
            return [text]

        for separator in self.separators:
            if separator in text:
                parts = text.split(separator, 1)
                left, right = parts[0], parts[1]
                if self._count_tokens(left) >= self.config['min_chunk_size']:
                    return (self._recursive_split_to_texts(left) +
                            self._recursive_split_to_texts(right))

        # 没有找到合适的分隔符，在中间切分
        mid = len(text) // 2
        return (self._recursive_split_to_texts(text[:mid]) +
                self._recursive_split_to_texts(text[mid:]))

    def chunk_to_vector_chunks(self, text: str, metadata: Dict = None) -> List:
        """
        分块并返回 VectorChunk 对象列表（用于向量化）

        Args:
            text: 输入文本
            metadata: 元数据

        Returns:
            VectorChunk 对象列表
        """
        from app.service.core.embedding.vector_types import VectorChunk

        chunk_texts = self.chunk_to_texts(text)
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


class SemanticChunker(BaseChunker):
    """语义分块策略"""

    def __init__(self, config: Optional[Dict] = None, embedding_model=None):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.similarity_threshold = config.get('similarity_threshold', 0.7) if config else 0.7

    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        sentence_delimiters = r'(?<=[。！？.!?])\s+'
        sentences = re.split(sentence_delimiters, text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarity(self, sent1: str, sent2: str) -> float:
        """计算两个句子的相似度"""
        if self.embedding_model:
            emb1 = self.embedding_model.encode(sent1)
            emb2 = self.embedding_model.encode(sent2)
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        else:
            words1 = set(re.findall(r'\w+', sent1.lower()))
            words2 = set(re.findall(r'\w+', sent2.lower()))
            if not words1 or not words2:
                return 0
            return len(words1 & words2) / len(words1 | words2)

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = sentences[0]
        chunk_id = 0

        for i in range(1, len(sentences)):
            similarity = self._compute_similarity(sentences[i - 1], sentences[i])

            if similarity < self.similarity_threshold and \
                    self._count_tokens(current_chunk) >= self.config['min_chunk_size']:
                chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                               metadata or {}, chunk_id)
                chunks.append(chunk_obj)
                current_chunk = sentences[i]
                chunk_id += 1
            else:
                current_chunk += sentences[i]

        if current_chunk:
            chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

        return chunks


class SentenceChunker(BaseChunker):
    """句子级分块策略"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.sentences_per_chunk = config.get('sentences_per_chunk', 5) if config else 5

    def _split_sentences(self, text: str) -> List[str]:
        sentence_delimiters = r'(?<=[。！？.!?])\s+'
        return re.split(sentence_delimiters, text)

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        sentences = self._split_sentences(text)
        chunks = []
        chunk_id = 0

        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk_text = ''.join(sentences[i:i + self.sentences_per_chunk])
            chunk_obj = self._create_chunk(chunk_text, 0, 0,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)
            chunk_id += 1

        return chunks


class ParagraphChunker(BaseChunker):
    """段落级分块策略"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

    def _split_paragraphs(self, text: str) -> List[str]:
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        paragraphs = self._split_paragraphs(text)
        chunks = []
        chunk_id = 0
        current_chunk = ""

        for para in paragraphs:
            if self._count_tokens(para) > self.config['chunk_token_num']:
                if current_chunk:
                    chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                                   metadata or {}, chunk_id)
                    chunks.append(chunk_obj)
                    chunk_id += 1
                    current_chunk = ""

                recursive_chunker = RecursiveChunker(self.config)
                sub_chunks = recursive_chunker.chunk(para, metadata)
                for sc in sub_chunks:
                    sc.id = f"chunk_{chunk_id}"
                    chunks.append(sc)
                    chunk_id += 1
            elif self._count_tokens(current_chunk + "\n" + para) > self.config['chunk_token_num']:
                if current_chunk:
                    chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                                   metadata or {}, chunk_id)
                    chunks.append(chunk_obj)
                    chunk_id += 1
                current_chunk = para
            else:
                current_chunk += "\n" + para if current_chunk else para

        if current_chunk:
            chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

        return chunks