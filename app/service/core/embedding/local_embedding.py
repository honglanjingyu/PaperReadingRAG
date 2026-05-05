# app/service/core/embedding/local_embedding.py

import os
import sys
from typing import List, Optional
import numpy as np
import logging

from .base_embedding import BaseEmbeddingModel

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers未安装，本地Embedding模型不可用")


def get_local_embedding_path() -> str:
    """获取本地 Embedding 模型路径"""
    return os.getenv("LOCAL_EMBEDDING_PATH")


def get_local_embedding_model_name() -> str:
    """获取本地 Embedding 模型名称"""
    return os.getenv("LOCAL_EMBEDDING_MODEL", "bge-small-zh")


class LocalEmbeddingModel(BaseEmbeddingModel):
    """本地Embedding模型 - 使用 sentence-transformers"""

    MODEL_PATHS = {
        "bge-small-zh": r"E:\HF_HOME\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620",
        "bge-base-zh": r"E:\HF_HOME\hub\models--BAAI--bge-base-zh-v1.5\snapshots\bge-base-zh-v1.5",
        "bge-large-zh": r"E:\HF_HOME\hub\models--BAAI--bge-large-zh-v1.5\snapshots\bge-large-zh-v1.5",
        "m3e-base": r"E:\HF_HOME\hub\models--moka-ai--m3e-base\snapshots\m3e-base",
        "text2vec-base": r"E:\HF_HOME\hub\models--shibing624--text2vec-base-chinese\snapshots\text2vec-base-chinese",
    }

    MODEL_DIMENSIONS = {
        "bge-small-zh": 512,
        "bge-base-zh": 768,
        "bge-large-zh": 1024,
        "m3e-base": 768,
        "text2vec-base": 768,
    }

    def __init__(
            self,
            model_name: str = None,
            model_path: str = None,
            device: str = "cpu",
            batch_size: int = 32,
            use_cache: bool = True
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        # 优先级：参数 > 环境变量 > 默认值
        self._model_key = (model_name or get_local_embedding_model_name()).lower()
        self._model_path = model_path or get_local_embedding_path() or self.MODEL_PATHS.get(self._model_key)

        if not self._model_path:
            raise ValueError(
                f"未知的模型: {self._model_key}，请设置 LOCAL_EMBEDDING_PATH 或 LOCAL_EMBEDDING_MODEL"
            )

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"模型路径不存在: {self._model_path}")

        self._device = device
        self._batch_size = batch_size
        self._use_cache = use_cache
        self._cache = {} if use_cache else None

        logger.info(f"加载本地Embedding模型: {self._model_key} from {self._model_path}")
        self._model = SentenceTransformer(self._model_path, device=device)

        self._dimension = self.MODEL_DIMENSIONS.get(self._model_key, 768)
        logger.info(f"本地模型加载完成: 维度={self._dimension}, 设备={device}")

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_key

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text:
            return None

        if self._use_cache and text in self._cache:
            return self._cache[text]

        try:
            embedding = self._model.encode(text, normalize_embeddings=True)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            if self._use_cache:
                self._cache[text] = embedding_list

            return embedding_list
        except Exception as e:
            logger.error(f"本地模型向量生成失败: {e}")
            return None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []

        if self._use_cache:
            results = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                if text in self._cache:
                    results.append(self._cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    results.append(None)

            if uncached_texts:
                try:
                    embeddings = self._model.encode(
                        uncached_texts,
                        batch_size=self._batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )

                    for idx, embedding in zip(uncached_indices, embeddings):
                        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                        results[idx] = embedding_list

                        if self._use_cache:
                            self._cache[texts[idx]] = embedding_list

                except Exception as e:
                    logger.error(f"批量向量生成失败: {e}")

            return results

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            results = []
            for embedding in embeddings:
                embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                results.append(embedding_list)

            return results

        except Exception as e:
            logger.error(f"批量向量生成失败: {e}")
            return [None] * len(texts)

    def clear_cache(self):
        if self._cache:
            self._cache.clear()
            logger.info("缓存已清除")

    def to_device(self, device: str):
        self._model.to(device)
        self._device = device
        logger.info(f"模型已切换到设备: {device}")