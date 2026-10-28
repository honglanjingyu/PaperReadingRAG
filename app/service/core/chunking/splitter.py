"""
文档分块器 - 父子分块策略（精简版）
"""

import hashlib
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from .types import ParentChildDocument, ParentChunk, ChildChunk

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
# 非 ASCII 且非汉字（U+4E00-U+9FFF 之外的全部字符），把 2/3/4 字节区一网打尽，
# 用来从「非 ASCII 字符数」里扣掉标点、全角、扩展区、假名、谚文、emoji 等非汉字
_NON_HAN_RE = re.compile(r'[\u0080-\u4dff\ua000-\uffff\U00010000-\U0010ffff]')


def _tokens_from_counts(cjk_count: int, other_count: int) -> int:
    """由中文字符数与其他字符数估算 token，与原公式 int(cjk/1.5 + other/4) 逐位一致"""
    if cjk_count <= 0 and other_count <= 0:
        return 0
    return int(cjk_count / 1.5 + other_count / 4)


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
        # token 估算的小文本 LRU 缓存（同一段文本在过滤、写 token_count、
        # overlap 校验里会被反复数，缓存能直接消掉这部分重复）
        self._token_cache: Dict[str, int] = {}
        # 字符计数缓存：_post_split_oversize、_apply_overlap、写 token_count
        # 都会对刚数过的同一段文本再数一次，按 text 缓存 (cjk, other) 结果
        self._char_cache: Dict[str, Tuple[int, int]] = {}

    # 字符计数 / token 估算共用的缓存阈值：父块 430~450 字要能覆盖（512 时
    # 正好卡在边缘，命中率上不去）
    _CACHE_MAX_TEXT = 1024
    _TOKEN_CACHE_MAX_ENTRIES = 4096
    _CHAR_CACHE_MAX_ENTRIES = 1024

    def _char_counts(self, text: str):
        """返回 (中文字符数, 其他字符数)，短文本走缓存

        汉字数不再逐字符 findall，改成「非 ASCII 字符数 - 非汉字字符数」，
        两次 C 层扫描即可，与 _CJK_RE.findall 严格等价。等价性：非 ASCII 字符里
        除 U+4E00-U+9FFF 外全部落在 _NON_HAN_RE 的区间内，两者互补，差集即汉字。
        对拍见 work/eval/scripts/bench_cjk_v3.py：固定 14 例 + 随机 300 例全部
        一致，30 万字 33.56ms -> 4.25ms（7.9x）。
        """
        if not text:
            return 0, 0
        cached = self._char_cache.get(text)
        if cached is not None:
            return cached
        if text.isascii():
            counts = (0, len(text))
        else:
            non_ascii = len(text) - len(text.encode("ascii", "ignore"))
            cjk = non_ascii - len(_NON_HAN_RE.findall(text))
            counts = (cjk, len(text) - cjk)
        if len(text) <= self._CACHE_MAX_TEXT:
            if len(self._char_cache) >= self._CHAR_CACHE_MAX_ENTRIES:
                self._char_cache.clear()
            self._char_cache[text] = counts
        return counts

    @staticmethod
    def _is_cjk(ch: str) -> bool:
        return '\u4e00' <= ch <= '\u9fff'

    def _count_tokens(self, text: str) -> int:
        """估算token数量（与原公式等价，短文本走缓存）"""
        if not text:
            return 0
        cached = self._token_cache.get(text)
        if cached is not None:
            return cached
        tokens = _tokens_from_counts(*self._char_counts(text))
        if len(text) <= self._CACHE_MAX_TEXT:
            if len(self._token_cache) >= self._TOKEN_CACHE_MAX_ENTRIES:
                self._token_cache.clear()
            self._token_cache[text] = tokens
        return tokens

    def _split_by_separator(
            self, text: str, max_size: int,
            text_counts: Optional[Tuple[int, int]] = None
    ) -> List[str]:
        """按分隔符分割文本

        text_counts 给定时直接用调用方已算好的 (cjk, other)，不再对 text
        整段重扫（父块切子块时由 split_document 传下来）。
        """
        if text_counts is not None:
            if _tokens_from_counts(*text_counts) <= max_size:
                return [text]
        elif self._count_tokens(text) <= max_size:
            return [text]

        for separator in self.separators:
            if separator in text:
                sep_cjk, sep_other = self._char_counts(separator)
                parts = text.split(separator)
                chunks = []
                current = ""
                # 维护 current 的字符计数，拼接时只做加法，不再全量重数
                cur_cjk = cur_other = 0

                for part in parts:
                    if not part.strip():
                        continue
                    part_cjk, part_other = self._char_counts(part)
                    # 单个 part 本身已超限时，继续拼接只会更超，先落盘再做二次切分
                    if _tokens_from_counts(part_cjk, part_other) > max_size:
                        if current:
                            chunks.append(current)
                            current = ""
                            cur_cjk = cur_other = 0
                        chunks.extend(self._split_by_length(part, max_size))
                        continue
                    if current:
                        new_cjk = cur_cjk + sep_cjk + part_cjk
                        new_other = cur_other + sep_other + part_other
                    else:
                        new_cjk, new_other = part_cjk, part_other
                    if _tokens_from_counts(new_cjk, new_other) <= max_size:
                        current = current + separator + part if current else part
                        cur_cjk, cur_other = new_cjk, new_other
                    else:
                        if current:
                            chunks.append(current)
                        current = part
                        cur_cjk, cur_other = part_cjk, part_other

                if current:
                    chunks.append(current)
                if len(chunks) > 1:
                    return self._post_split_oversize(chunks, max_size)

        return self._split_by_length(text, max_size)

    def _post_split_oversize(self, chunks: List[str], max_size: int) -> List[str]:
        """兜底：把仍然超过 max_size 的块再按长度切一次"""
        result = []
        for chunk in chunks:
            if self._count_tokens(chunk) <= max_size:
                result.append(chunk)
            else:
                result.extend(self._split_by_length(chunk, max_size))
        return result

    # 纯 CJK 时 1 个字折算的 token 数，用来给定长粗切定上界
    _CJK_TOKENS_PER_CHAR = 1.5

    def _split_by_length(self, text: str, max_size: int) -> List[str]:
        """按长度切分：定长粗切 + 边界微调

        旧实现逐字推进（O(n) 但每个字一次 Python 循环，纯 Python 里 0.13 μs/字）。
        新实现先按「全是汉字」的最坏情况定长切：token 公式 int(cjk/1.5+other/4)
        下 cjk/1.5 <= max_size，所以一块最多 int(1.5*max_size)+1 个字，按这个步长
        粗切后每块只需一次 _char_counts（两次 C 层扫描）再微调边界：

          1) 超限就从尾部删字（纯汉字时最多删 1 个）；
          2) 还有余量就按本块中英密度估算右扩，最多 3 轮；
          3) 逐字收尾，直到再加一个字就超限。

        微调后每块都是「再加一个字就超限」的最大前缀，因此切分点与逐字版
        **完全一致**，输出逐块不变。
        """
        n = len(text)
        if n == 0:
            return []
        # 纯 CJK 时一块能放下的最大字数；+1 是 int() 截断的余量
        step = int(self._CJK_TOKENS_PER_CHAR * max_size) + 1
        if step < 1:
            step = 1
        # q = 8*cjk + 3*other 是 token 数的 12 倍分子，int(q/12) <= max_size
        # 等价于 q <= 12*max_size + 11
        limit_q = 12 * max_size + 11
        chunks = []
        start = 0
        while start < n:
            end = min(start + step, n)
            cjk, other = self._char_counts(text[start:end])
            # 1) 超限就收缩（纯 CJK 时最多收缩 1 个字）
            while end > start + 1 and 8 * cjk + 3 * other > limit_q:
                end -= 1
                if self._is_cjk(text[end]):
                    cjk -= 1
                else:
                    other -= 1
            # 2) 还有余量就按本块中英密度估算右扩，最多 3 轮
            for _ in range(3):
                if end >= n:
                    break
                q = 8 * cjk + 3 * other
                length = end - start
                avg_cost = max(3, q // length) if length else 3
                extra = (limit_q - q) // avg_cost
                if extra <= 0:
                    break
                new_end = min(end + extra, n)
                seg_cjk, seg_other = self._char_counts(text[end:new_end])
                cjk += seg_cjk
                other += seg_other
                end = new_end
                while end > start + 1 and 8 * cjk + 3 * other > limit_q:
                    end -= 1
                    if self._is_cjk(text[end]):
                        cjk -= 1
                    else:
                        other -= 1
            # 3) 逐字收尾：扩到再加一个字就超限为止（残量已被前两步压到很小）
            while end < n:
                if self._is_cjk(text[end]):
                    nc, no = cjk + 1, other
                else:
                    nc, no = cjk, other + 1
                if 8 * nc + 3 * no > limit_q:
                    break
                cjk, other = nc, no
                end += 1
            chunks.append(text[start:end])
            start = end
        return chunks

    def _overlap_token_budget(self, overlap: int) -> int:
        """把 overlap 的字符数换算成保守的 token 预算（中文最费 token，按 1 字 1.5 token 算）"""
        if overlap <= 0:
            return 0
        return int(overlap * 4 / 1.5) + 1

    def _apply_overlap(self, chunks: List[str], overlap: int, max_size: int = None) -> List[str]:
        """应用重叠

        max_size 给定时逐字校验：叠加后的块长仍不超过 max_size，放不下的尾部
        字符一律截掉，不再靠调用方预留预算兜底。
        """
        if overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-min(overlap * 4, len(prev_chunk)):]
                if max_size is not None:
                    # 从尾部往前收，直到 前缀+chunk 的 token 数落在上限内；
                    # 用增量计数代替每次重数字符串，避免收缩过程退化成 O(len²)
                    chunk_cjk, chunk_other = self._char_counts(chunk)
                    ov_cjk, ov_other = self._char_counts(overlap_text)
                    while overlap_text and _tokens_from_counts(
                            ov_cjk + chunk_cjk, ov_other + chunk_other) > max_size:
                        drop = overlap_text[0]
                        if self._is_cjk(drop):
                            ov_cjk -= 1
                        else:
                            ov_other -= 1
                        overlap_text = overlap_text[1:]
                    if not overlap_text:
                        overlapped.append(chunk)
                        continue
                chunk = overlap_text + chunk
            overlapped.append(chunk)
        return overlapped

    def split_to_parents(self, text: str) -> List[str]:
        """切分为父块"""
        if not text:
            return []
        # 先给 overlap 预留 token 预算，叠加后总长才不会越过 parent_chunk_size
        budget = self._overlap_token_budget(self.parent_overlap)
        max_size = max(1, self.parent_chunk_size - budget)
        chunks = self._split_by_separator(text, max_size)
        return self._apply_overlap(chunks, self.parent_overlap, self.parent_chunk_size)

    def split_parent_to_children(
            self, parent_text: str, parent_id: str,
            parent_counts: Optional[Tuple[int, int]] = None
    ) -> List[ChildChunk]:
        """将父块切分为子块

        parent_counts 是父文本的 (cjk, other) 计数。split_document 已经算过，
        传下来后子块切分就不必对父文本整段再做一次正则扫描。
        """
        if not parent_text:
            return []
        # 子块按完整 child_chunk_size 切，overlap 由 _apply_overlap 逐字收缩，
        # 避免扣预算后块长低于 min_child_size 被子块过滤掉
        child_texts = self._split_by_separator(
            parent_text, self.child_chunk_size, parent_counts)
        child_texts = self._apply_overlap(child_texts, self.child_overlap, self.child_chunk_size)
        children = []
        for child_text in child_texts:
            token_count = self._count_tokens(child_text)
            if token_count < self.min_child_size:
                continue
            i = len(children)
            child_hash = hashlib.md5(f"{parent_id}_{i}_{child_text[:100]}".encode()).hexdigest()[:16]
            children.append(ChildChunk(
                id=f"child_{child_hash}",
                content=child_text,
                parent_id=parent_id,
                chunk_index=i,
                token_count=token_count
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

            # 父块计数只算这一次：既用于父块 token_count，也传给子块切分做
            # 入口判定，不再让 split_parent_to_children 整段重扫
            parent_counts = self._char_counts(parent_text)
            parent_token_count = _tokens_from_counts(*parent_counts)

            children = self.split_parent_to_children(parent_text, parent_id, parent_counts)

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
                token_count=parent_token_count,
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
    """简单分块（兼容原函数）

    语义就是「按块长切」：每块字符数不超过 chunk_size，块之间不重叠，
    拼接后与原文本逐字一致。不走 split_to_parents 的 overlap 逻辑。
    """
    if not text:
        return []
    chunk_size = max(1, int(chunk_size))
    if len(text) <= chunk_size:
        return [text]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


__all__ = [
    'ParentChildSplitter',
    'chunk_document_parent_child',
    'chunk_text_simple',
]