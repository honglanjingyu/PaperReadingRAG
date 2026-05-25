"""多模态数据模型"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class MediaType(str, Enum):
    """媒体类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    PDF = "pdf"
    DOCX = "docx"


class ExtractStatus(str, Enum):
    """提取状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExtractedContent:
    """提取的内容"""
    media_type: MediaType
    original_filename: str
    text_content: str = ""  # 提取的文字内容
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ExtractStatus = ExtractStatus.PENDING
    error: Optional[str] = None

    # 图片特有
    ocr_confidence: float = 0.0
    detected_language: str = "zh"

    # 音频/视频特有
    duration_seconds: float = 0.0
    transcript_confidence: float = 0.0
    speaker_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "media_type": self.media_type.value,
            "original_filename": self.original_filename,
            "text_content": self.text_content,
            "metadata": self.metadata,
            "status": self.status.value,
            "error": self.error,
            "ocr_confidence": self.ocr_confidence,
            "detected_language": self.detected_language,
            "duration_seconds": self.duration_seconds,
            "transcript_confidence": self.transcript_confidence
        }


@dataclass
class MultimodalChunk:
    """多模态分块（继承原有 VectorChunk）"""
    id: str
    content: str  # 提取的文字内容
    media_type: MediaType
    original_filename: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 原有字段（兼容）
    vector: List[float] = field(default_factory=list)
    token_count: int = 0
    user_level: str = "normal"
    parent_id: str = ""
    parent_content: str = ""

    def to_vector_chunk(self):
        """转换为原有的 VectorChunk 格式"""
        from app.service.core.embedding import VectorChunk

        return VectorChunk(
            id=self.id,
            content=self.content,
            vector=self.vector,
            metadata={
                **self.metadata,
                "media_type": self.media_type.value,
                "original_filename": self.original_filename,
                "chunk_index": self.chunk_index
            },
            token_count=self.token_count,
            user_level=self.user_level,
            parent_id=self.parent_id,
            parent_content=self.parent_content
        )