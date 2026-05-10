# app/service/core/embedding/vector_types.py - 更新
"""向量相关数据结构（支持用户等级）"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class VectorChunk:
    """带向量的分块数据结构"""
    id: str
    content: str
    vector: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    chunk_index: int = 0
    user_level: str = "normal"  # 文档所属用户等级：normal, admin, owner

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'content': self.content,
            'vector': self.vector[:10] if self.vector else [],
            'vector_dim': len(self.vector),
            'metadata': self.metadata,
            'token_count': self.token_count,
            'chunk_index': self.chunk_index,
            'user_level': self.user_level
        }

    def to_es_document(self, kb_id: str = None, doc_name: str = None) -> Dict:
        """转换为Elasticsearch文档格式"""
        doc = {
            "id": self.id,
            "content_with_weight": self.content,
            "metadata": self.metadata,
            "token_count": self.token_count,
            "created_at": datetime.now().isoformat(),
            "user_level": self.user_level  # 添加用户等级字段
        }

        if self.vector:
            doc[f"q_{len(self.vector)}_vec"] = self.vector

        if kb_id:
            doc["kb_id"] = kb_id
        if doc_name:
            doc["docnm_kwd"] = doc_name

        return doc


__all__ = ['VectorChunk']