"""
分块策略实现
支持多种分块方式：固定长度、语义、递归、句子、段落
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ChunkStrategy(Enum):
    """分块策略枚举"""
    FIXED_TOKEN = "fixed_token"  # 固定 token 数
    SEMANTIC = "semantic"  # 语义分块
    RECURSIVE = "recursive"  # 递归分块
    SENTENCE = "sentence"  # 句子级分块
    PARAGRAPH = "paragraph"  # 段落级分块


@dataclass
class Chunk:
    """分块数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    start_idx: int
    end_idx: int
    token_count: int = 0


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
        """估算 token 数量（简化版，实际项目中可使用 tiktoken）"""
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
    """
    固定 Token 数分块策略
    按指定的 token 数进行切分，支持重叠
    """

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
            # 获取当前块
            end = min(start + self.chunk_size * 4, length)  # 先用字符估算

            # 在分隔符处切分
            chunk_text = text[start:end]
            chunk_obj = self._create_chunk(chunk_text, start, end,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

            # 移动起始位置（考虑重叠）
            start = end - self.overlap * 4
            chunk_id += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """
    递归分块策略
    优先在段落、句子、短语等自然边界处切分
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.separators = [
            "\n\n",  # 段落
            "\n",  # 换行
            "。", "！", "？", "；",  # 中文句子分隔符
            ". ", "! ", "? ", "; ",  # 英文句子分隔符
            "，", ", ",  # 逗号
            " ",  # 空格
        ]

    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        if not text:
            return []

        chunks = []
        current_chunk = ""
        chunk_id = 0

        for separator in self.separators:
            if len(current_chunk) < self.config['chunk_token_num'] * 4:
                # 尝试用当前分隔符切分
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
                # 当前块已满，递归处理剩余部分
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

        # 添加最后一个块
        if current_chunk:
            chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

        return chunks


class SemanticChunker(BaseChunker):
    """
    语义分块策略
    基于句子嵌入相似度进行分块
    """

    def __init__(self, config: Optional[Dict] = None, embedding_model=None):
        super().__init__(config)
        self.embedding_model = embedding_model
        self.similarity_threshold = config.get('similarity_threshold', 0.7) if config else 0.7

    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 句子分隔符正则
        sentence_delimiters = r'(?<=[。！？.!?])\s+'
        sentences = re.split(sentence_delimiters, text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarity(self, sent1: str, sent2: str) -> float:
        """计算两个句子的相似度"""
        if self.embedding_model:
            # 使用嵌入模型计算相似度
            emb1 = self.embedding_model.encode(sent1)
            emb2 = self.embedding_model.encode(sent2)
            # 余弦相似度
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        else:
            # 简化版：基于共同词汇的 Jaccard 相似度
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

            # 如果相似度低于阈值，且当前块已达到最小大小，则切分
            if similarity < self.similarity_threshold and \
                    self._count_tokens(current_chunk) >= self.config['min_chunk_size']:
                chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                               metadata or {}, chunk_id)
                chunks.append(chunk_obj)
                current_chunk = sentences[i]
                chunk_id += 1
            else:
                # 合并到当前块
                current_chunk += sentences[i]

        # 添加最后一个块
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
        """将文本分割成句子"""
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
        """按段落分割"""
        # 按连续换行分割
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
            # 如果单个段落超过限制，需要进一步切分
            if self._count_tokens(para) > self.config['chunk_token_num']:
                if current_chunk:
                    chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                                   metadata or {}, chunk_id)
                    chunks.append(chunk_obj)
                    chunk_id += 1
                    current_chunk = ""

                # 使用递归分块处理长段落
                recursive_chunker = RecursiveChunker(self.config)
                sub_chunks = recursive_chunker.chunk(para, metadata)
                for sc in sub_chunks:
                    sc.id = f"chunk_{chunk_id}"
                    chunks.append(sc)
                    chunk_id += 1
            elif self._count_tokens(current_chunk + "\n" + para) > self.config['chunk_token_num']:
                # 当前块已满
                if current_chunk:
                    chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                                   metadata or {}, chunk_id)
                    chunks.append(chunk_obj)
                    chunk_id += 1
                current_chunk = para
            else:
                # 合并到当前块
                current_chunk += "\n" + para if current_chunk else para

        # 添加最后一个块
        if current_chunk:
            chunk_obj = self._create_chunk(current_chunk, 0, 0,
                                           metadata or {}, chunk_id)
            chunks.append(chunk_obj)

        return chunks