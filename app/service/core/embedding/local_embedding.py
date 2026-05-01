"""
本地Embedding模型
使用sentence-transformers加载本地模型
"""

import os
import sys
from typing import List, Optional
import numpy as np
import logging

from .base_embedding import BaseEmbeddingModel

logger = logging.getLogger(__name__)

# 尝试导入 sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers未安装，本地Embedding模型不可用")


class LocalEmbeddingModel(BaseEmbeddingModel):
    """
    本地Embedding模型
    使用sentence-transformers加载本地模型
    """

    # 支持的本地模型路径映射
    MODEL_PATHS = {
        "bge-small-zh": r"E:\HF_HOME\hub\models--BAAI--bge-small-zh-v1.5\snapshots\7999e1d3359715c523056ef9478215996d62a620",
        "bge-base-zh": r"E:\HF_HOME\hub\models--BAAI--bge-base-zh-v1.5\snapshots\bge-base-zh-v1.5",
        "bge-large-zh": r"E:\HF_HOME\hub\models--BAAI--bge-large-zh-v1.5\snapshots\bge-large-zh-v1.5",
        "m3e-base": r"E:\HF_HOME\hub\models--moka-ai--m3e-base\snapshots\m3e-base",
        "text2vec-base": r"E:\HF_HOME\hub\models--shibing624--text2vec-base-chinese\snapshots\text2vec-base-chinese",
    }

    # 模型维度映射
    MODEL_DIMENSIONS = {
        "bge-small-zh": 512,
        "bge-base-zh": 768,
        "bge-large-zh": 1024,
        "m3e-base": 768,
        "text2vec-base": 768,
    }

    def __init__(
            self,
            model_name: str = "bge-small-zh",
            model_path: str = None,
            device: str = "cpu",
            batch_size: int = 32,
            use_cache: bool = True
    ):
        """
        初始化本地Embedding模型

        Args:
            model_name: 模型名称（如 bge-small-zh, bge-base-zh, m3e-base）
            model_path: 模型路径（可选，如果不指定则使用预设路径）
            device: 运行设备 (cpu/cuda)
            batch_size: 批处理大小
            use_cache: 是否使用缓存
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")

        self._model_key = model_name.lower()
        self._model_path = model_path or self.MODEL_PATHS.get(self._model_key)

        if not self._model_path:
            raise ValueError(
                f"未知的模型: {model_name}，请指定 model_path 或使用预设模型: {list(self.MODEL_PATHS.keys())}")

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"模型路径不存在: {self._model_path}")

        self._device = device
        self._batch_size = batch_size
        self._use_cache = use_cache
        self._cache = {} if use_cache else None

        # 加载模型
        logger.info(f"加载本地Embedding模型: {self._model_key} from {self._model_path}")
        self._model = SentenceTransformer(self._model_path, device=device)

        # 获取向量维度
        self._dimension = self.MODEL_DIMENSIONS.get(self._model_key, 768)

        logger.info(f"本地模型加载完成: 维度={self._dimension}, 设备={device}")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成单个文本的向量"""
        if not text:
            return None

        # 检查缓存
        if self._use_cache and text in self._cache:
            return self._cache[text]

        try:
            embedding = self._model.encode(text, normalize_embeddings=True)
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            # 缓存结果
            if self._use_cache:
                self._cache[text] = embedding_list

            return embedding_list
        except Exception as e:
            logger.error(f"本地模型向量生成失败: {e}")
            return None

    def generate_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量生成文本向量"""
        if not texts:
            return []

        # 检查缓存
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
                    results.append(None)  # 占位

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

                        # 缓存
                        if self._use_cache:
                            self._cache[texts[idx]] = embedding_list

                except Exception as e:
                    logger.error(f"批量向量生成失败: {e}")
                    # 失败的位置保持 None

            return results

        # 不使用缓存，直接批量生成
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

    def get_embedding_dimension(self) -> int:
        """获取模型向量维度"""
        return self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_key

    def clear_cache(self):
        """清除缓存"""
        if self._cache:
            self._cache.clear()
            logger.info("缓存已清除")

    def to_device(self, device: str):
        """切换设备"""
        self._model.to(device)
        self._device = device
        logger.info(f"模型已切换到设备: {device}")