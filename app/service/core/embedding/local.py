"""本地 Embedding 模型（sentence-transformers）"""

import os
import logging
from typing import List, Optional

from .base import BaseEmbeddingModel

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_AVAILABLE = True
except ImportError:
    SENTENCE_AVAILABLE = False
    logger.warning("sentence-transformers 未安装，本地模型不可用")


def _get_model_path(model_key: str) -> str:
    """获取本地模型路径"""
    paths = {
        "bge-small-zh": r"E:\HF_HOME\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620",
        "bge-base-zh": r"E:\HF_HOME\hub\models--BAAI--bge-base-zh-v1.5\snapshots\bge-base-zh-v1.5",
        "bge-large-zh": r"E:\HF_HOME\hub\models--BAAI--bge-large-zh-v1.5\snapshots\bge-large-zh-v1.5",
    }
    return paths.get(model_key, "")


def _get_model_name() -> str:
    return os.getenv("LOCAL_EMBEDDING_MODEL", "bge-small-zh")


def _get_model_path_from_env() -> str:
    return os.getenv("LOCAL_EMBEDDING_PATH", "")


class LocalEmbeddingModel(BaseEmbeddingModel):
    """本地 Embedding 模型"""

    MODEL_DIMS = {
        "bge-small-zh": 512,
        "bge-base-zh": 768,
        "bge-large-zh": 1024,
        "m3e-base": 768,
    }

    def __init__(self, model_name: str = None, device: str = "cpu", batch_size: int = 32):
        if not SENTENCE_AVAILABLE:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        self._model_key = (model_name or _get_model_name()).lower()
        self._model_path = _get_model_path_from_env() or _get_model_path(self._model_key)

        if not self._model_path:
            raise ValueError(f"未知模型: {self._model_key}，请设置 LOCAL_EMBEDDING_PATH")

        logger.info(f"加载本地模型: {self._model_key} from {self._model_path}")
        self._model = SentenceTransformer(self._model_path, device=device)
        self._device = device
        self._batch_size = batch_size
        self._dimension = self.MODEL_DIMS.get(self._model_key, 768)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_key

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text:
            return None
        try:
            emb = self._model.encode(text, normalize_embeddings=True)
            return emb.tolist() if hasattr(emb, 'tolist') else list(emb)
        except Exception as e:
            logger.error(f"本地模型生成失败: {e}")
            return None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []
        try:
            embeddings = self._model.encode(
                texts, batch_size=self._batch_size,
                normalize_embeddings=True, show_progress_bar=False
            )
            return [
                emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                for emb in embeddings
            ]
        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            return [None] * len(texts)