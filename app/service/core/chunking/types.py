"""
分块数据结构定义
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """通用分块数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    start_idx: int = 0
    end_idx: int = 0
    token_count: int = 0


@dataclass
class ChildChunk:
    """子块 - 用于检索"""
    id: str
    content: str
    parent_id: str
    chunk_index: int
    start_pos: int = 0
    end_pos: int = 0
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


__all__ = [
    'Chunk',
    'ChildChunk',
    'ParentChunk',
    'ParentChildDocument',
]