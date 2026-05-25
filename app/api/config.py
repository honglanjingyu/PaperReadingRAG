# app/api/config.py

import os
from pathlib import Path
from typing import Dict, Any

# 上传文件保存目录
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Web静态文件目录
WEB_DIR = Path(__file__).parent.parent / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)

# ========== 扩展支持的文件类型 ==========
# 原有文档类型
DOCUMENT_EXTENSIONS = {
    '.pdf': 'pdf',
    '.docx': 'docx',
    '.txt': 'text',
    '.md': 'text',
    '.markdown': 'text',
}

# 新增图片类型
IMAGE_EXTENSIONS = {
    '.jpg': 'image',
    '.jpeg': 'image',
    '.png': 'image',
    '.gif': 'image',
    '.bmp': 'image',
    '.webp': 'image',
    '.tiff': 'image',
    '.tif': 'image',
}

# 新增音频类型
AUDIO_EXTENSIONS = {
    '.mp3': 'audio',
    '.wav': 'audio',
    '.flac': 'audio',
    '.m4a': 'audio',
    '.aac': 'audio',
    '.ogg': 'audio',
}

# 新增视频类型
VIDEO_EXTENSIONS = {
    '.mp4': 'video',
    '.avi': 'video',
    '.mov': 'video',
    '.mkv': 'video',
    '.flv': 'video',
    '.wmv': 'video',
    '.webm': 'video',
}

# 合并所有支持的类型
SUPPORTED_EXTENSIONS = {
    **DOCUMENT_EXTENSIONS,
    **IMAGE_EXTENSIONS,
    **AUDIO_EXTENSIONS,
    **VIDEO_EXTENSIONS,
}

# 媒体类型映射
MEDIA_TYPE_MAP = {
    **{ext: 'document' for ext in DOCUMENT_EXTENSIONS},
    **{ext: 'image' for ext in IMAGE_EXTENSIONS},
    **{ext: 'audio' for ext in AUDIO_EXTENSIONS},
    **{ext: 'video' for ext in VIDEO_EXTENSIONS},
}


class Settings:
    """配置类"""

    # ... 原有配置保持不变 ...

    @property
    def chunk_size(self) -> int:
        return int(os.getenv("CHUNK_SIZE", "256"))

    @property
    def max_pages(self) -> int:
        return int(os.getenv("MAX_PAGES", "100000"))

    @property
    def index_name(self) -> str:
        return os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    @property
    def rerank_top_k(self) -> int:
        return int(os.getenv("RERANK_TOP_K", "5"))

    @property
    def similarity_top_k(self) -> int:
        return int(os.getenv("SIMILARITY_TOP_K", "10"))

    @property
    def similarity_threshold(self) -> float:
        return float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    @property
    def keyword_weight(self) -> float:
        return float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4"))

    @property
    def vector_weight(self) -> float:
        return float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))

    @property
    def rerank_type(self) -> str:
        return os.getenv("RERANK_TYPE", "remote")

    @property
    def enable_rerank(self) -> bool:
        return os.getenv("ENABLE_RERANK", "true").lower() == "true"

    @property
    def enable_query_rewrite(self) -> bool:
        return os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"


settings = Settings()

__all__ = ['settings', 'UPLOAD_DIR', 'WEB_DIR', 'SUPPORTED_EXTENSIONS', 'MEDIA_TYPE_MAP']