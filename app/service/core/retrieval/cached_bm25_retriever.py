# app/service/core/retrieval/cached_bm25_retriever.py

import pickle
import logging
from typing import List, Dict, Any, Optional

from .bm25_retriever import BM25Retriever, BM25Variant, create_bm25_retriever
from ..cache import get_cache_manager

logger = logging.getLogger(__name__)


class CachedBM25Retriever:
    """带缓存的 BM25 检索器 - 缓存构建好的 BM25 模型"""

    def __init__(
            self,
            variant: BM25Variant = BM25Variant.PLUS,
            use_jieba: bool = True,
            use_synonyms: bool = True,
            cache_ttl: int = None
    ):
        import os

        self.cache_manager = get_cache_manager()
        self.cache_ttl = cache_ttl or int(os.getenv("CACHE_BM25_TTL", "3600"))

        self.variant = variant
        self.use_jieba = use_jieba
        self.use_synonyms = use_synonyms

        self._local_cache: Dict[str, BM25Retriever] = {}

        # 添加调试日志
        logger.info(f"CachedBM25Retriever 初始化: variant={variant.value}, TTL={self.cache_ttl}s")

    def _get_cache_key(self, index_name: str) -> str:
        """生成缓存的索引 key"""
        return f"bm25:{index_name}:{self.variant.value}"

    def get_or_build(
            self,
            index_name: str,
            documents: List[Dict] = None,
            content_field: str = "content_with_weight",
            force_rebuild: bool = False
    ) -> Optional[BM25Retriever]:
        import time
        start = time.time()

        cache_key = self._get_cache_key(index_name)
        logger.info(f"BM25 缓存 key: {cache_key}")

        # 1. 检查本地内存缓存
        if not force_rebuild and cache_key in self._local_cache:
            logger.info(f"BM25 本地缓存命中: {index_name}, 耗时: {(time.time() - start) * 1000:.2f}ms")
            return self._local_cache[cache_key]

        # 2. 检查 Redis 缓存
        if not force_rebuild:
            logger.info(f"尝试从 Redis 读取 BM25 缓存: {cache_key}")
            cached_model = self.cache_manager.get("bm25", cache_key)
            if cached_model:
                logger.info(f"从 Redis 获取到缓存数据，长度: {len(cached_model)} 字符")
                try:
                    model = self._deserialize_model(cached_model)
                    if model:
                        self._local_cache[cache_key] = model
                        logger.info(f"BM25 Redis 缓存命中: {index_name}, 耗时: {(time.time() - start) * 1000:.2f}ms")
                        return model
                    else:
                        logger.warning(f"反序列化失败，返回 None")
                except Exception as e:
                    logger.warning(f"反序列化 BM25 模型失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.info(f"Redis 中未找到 BM25 缓存: {cache_key}")

        # 3. 构建新模型
        if not documents:
            logger.warning(f"未提供文档，无法构建 BM25 模型: {index_name}")
            return None

        logger.info(f"构建 BM25 模型: {index_name}, 文档数={len(documents)}")

        retriever = create_bm25_retriever(
            variant=self.variant.value,
            use_jieba=self.use_jieba,
            use_synonyms=self.use_synonyms
        )

        retriever.build_corpus(documents, content_field)

        if retriever.get_document_count() == 0:
            logger.warning(f"BM25 模型构建失败: 无有效文档")
            return None

        self._local_cache[cache_key] = retriever
        logger.info(f"BM25 模型已存入本地缓存")

        serialized = self._serialize_model(retriever)
        if serialized:
            logger.info(f"序列化成功，数据长度: {len(serialized)} 字符")
            result = self.cache_manager.set("bm25", cache_key, serialized, self.cache_ttl)
            logger.info(
                f"写入 Redis 结果: {result}, BM25 模型已缓存: {index_name}, 耗时: {(time.time() - start) * 1000:.2f}ms")
        else:
            logger.error(f"序列化失败，未缓存")

        return retriever

    # app/service/core/retrieval/cached_bm25_retriever.py

    def _serialize_model(self, retriever: BM25Retriever) -> Optional[str]:
        """序列化 BM25 模型 - 只缓存必要数据"""
        try:
            # 只缓存核心数据，不缓存模型对象
            model_data = {
                'corpus': retriever._corpus,
                'tokenized_corpus': retriever._tokenized_corpus,
                'variant': retriever.variant.value,
                'k1': retriever.k1,
                'b': retriever.b,
                'delta': retriever.delta
            }
            import pickle
            import zlib  # 添加压缩
            serialized = pickle.dumps(model_data)
            compressed = zlib.compress(serialized)
            logger.info(f"序列化成功，原始大小: {len(serialized)} bytes, 压缩后: {len(compressed)} bytes")
            return compressed.hex()
        except Exception as e:
            logger.error(f"序列化 BM25 模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _deserialize_model(self, data: str) -> Optional[BM25Retriever]:
        """反序列化 BM25 模型"""
        try:
            import pickle
            import zlib
            import time
            start = time.time()

            compressed = bytes.fromhex(data)
            logger.info(f"反序列化: 压缩数据大小 {len(compressed)} bytes")
            binary_data = zlib.decompress(compressed)
            logger.info(f"解压后大小: {len(binary_data)} bytes")

            model_data = pickle.loads(binary_data)
            logger.info(f"pickle 反序列化成功，耗时: {(time.time() - start) * 1000:.2f}ms")

            retriever = create_bm25_retriever(
                variant=model_data['variant'],
                use_jieba=self.use_jieba,
                use_synonyms=self.use_synonyms
            )
            retriever._corpus = model_data['corpus']
            retriever._tokenized_corpus = model_data['tokenized_corpus']
            retriever.k1 = model_data['k1']
            retriever.b = model_data['b']
            retriever.delta = model_data['delta']

            from rank_bm25 import BM25Okapi, BM25Plus, BM25L

            variant = BM25Variant(model_data['variant'])
            if variant == BM25Variant.PLUS:
                retriever._bm25_model = BM25Plus(
                    retriever._tokenized_corpus,
                    k1=retriever.k1,
                    b=retriever.b,
                    delta=retriever.delta
                )
            elif variant == BM25Variant.L:
                retriever._bm25_model = BM25L(
                    retriever._tokenized_corpus,
                    k1=retriever.k1,
                    b=retriever.b
                )
            else:
                retriever._bm25_model = BM25Okapi(
                    retriever._tokenized_corpus,
                    k1=retriever.k1,
                    b=retriever.b
                )

            logger.info(f"BM25 模型重建成功，文档数: {len(retriever._corpus)}")
            return retriever
        except Exception as e:
            logger.error(f"反序列化 BM25 模型失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def invalidate_cache(self, index_name: str):
        cache_key = self._get_cache_key(index_name)
        self._local_cache.pop(cache_key, None)
        self.cache_manager.delete("bm25", cache_key)
        logger.info(f"BM25 缓存已失效: {index_name}")

    def search(
            self,
            query: str,
            index_name: str,
            documents: List[Dict] = None,
            top_k: int = 5,
            content_field: str = "content_with_weight",
            **kwargs
    ) -> List[Dict]:
        import time
        start = time.time()

        retriever = self.get_or_build(index_name, documents, content_field)

        if not retriever:
            return []

        result = retriever.search(
            query=query,
            top_k=top_k,
            content_field=content_field,
            **kwargs
        )

        logger.info(
            f"BM25 搜索完成: query={query[:30]}..., 结果数={len(result)}, 总耗时: {(time.time() - start) * 1000:.2f}ms")
        return result


__all__ = ['CachedBM25Retriever']