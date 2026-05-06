# app/api/config.py
"""
API配置模块
"""

import os
from pathlib import Path
from typing import Dict, Any

# 上传文件保存目录
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Web静态文件目录
WEB_DIR = Path(__file__).parent.parent / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)

# 支持的文件类型
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.docx': 'docx',
    '.txt': 'text',
    '.md': 'text',
    '.markdown': 'text'
}


class Settings:
    """配置类"""

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


# 处理状态存储（简单内存存储，生产环境应使用Redis）
processing_status: Dict[str, Any] = {}


__all__ = ['settings', 'UPLOAD_DIR', 'WEB_DIR', 'SUPPORTED_EXTENSIONS', 'processing_status']