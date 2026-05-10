# app/service/core/chunking/chunk_types.py
"""分块数据结构定义 - 避免循环导入"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Chunk:
    """分块数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    start_idx: int
    end_idx: int
    token_count: int = 0