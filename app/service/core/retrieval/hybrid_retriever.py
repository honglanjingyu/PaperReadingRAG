# app/service/core/retrieval/hybrid_retriever.py

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器 - 融合 BM25 关键词检索和向量检索"""

    def __init__(
            self,
            vector_store=None,
            embedding_manager=None,
            use_jieba: bool = True,
            use_synonyms: bool = True,
            bm25_variant: str = "plus"
    ):
        """
        初始化混合检索器

        Args:
            vector_store: 向量存储实例
            embedding_manager: EmbeddingManager 实例
            use_jieba: 是否使用 jieba 分词
            use_synonyms: 是否使用同义词扩展
            bm25_variant: BM25 变体 ("okapi", "plus", "l")
        """
        from app.service.core.vector_store import get_vector_store
        from app.service.core.embedding import get_embedding_manager
        from .bm25_retriever import create_bm25_retriever

        self.vector_store = vector_store or get_vector_store()
        self.embedding_manager = embedding_manager or get_embedding_manager()

        # 初始化 BM25 检索器（支持同义词和 OR 语法）
        self.bm25_retriever = create_bm25_retriever(
            variant=bm25_variant,
            use_jieba=use_jieba,
            use_synonyms=use_synonyms
        )

        # 从环境变量读取混合检索权重
        # 权重优先级：参数 > 环境变量 > 默认值
        self._load_weights_from_env()

    def _load_weights_from_env(self):
        """从环境变量加载混合检索权重"""
        # 读取向量权重
        vector_weight_str = os.getenv("HYBRID_VECTOR_WEIGHT", "0.6")
        try:
            self.vector_weight = float(vector_weight_str)
        except ValueError:
            logger.warning(f"HYBRID_VECTOR_WEIGHT 值无效: {vector_weight_str}，使用默认值 0.6")
            self.vector_weight = 0.6

        # 读取关键词权重
        keyword_weight_str = os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4")
        try:
            self.keyword_weight = float(keyword_weight_str)
        except ValueError:
            logger.warning(f"HYBRID_KEYWORD_WEIGHT 值无效: {keyword_weight_str}，使用默认值 0.4")
            self.keyword_weight = 0.4

        # 验证权重和是否为1（如果不等，进行归一化）
        total = self.vector_weight + self.keyword_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"权重和不为1: vector={self.vector_weight}, keyword={self.keyword_weight}, total={total}")
            logger.info(f"正在进行归一化...")
            self.vector_weight = self.vector_weight / total
            self.keyword_weight = self.keyword_weight / total
            logger.info(f"归一化后: vector={self.vector_weight:.4f}, keyword={self.keyword_weight:.4f}")

        logger.info(f"混合检索权重: 向量={self.vector_weight}, 关键词={self.keyword_weight}")

    def set_weights(self, vector_weight: float, keyword_weight: float):
        """
        手动设置混合检索权重

        Args:
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        logger.info(f"混合检索权重已更新: 向量={self.vector_weight}, 关键词={self.keyword_weight}")

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重配置"""
        return {
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight
        }

    def hybrid_search(
            self,
            query: str,
            index_name: str,
            top_k: int = 5,
            keyword_weight: float = None,
            vector_weight: float = None,
            similarity_threshold: float = 0.3,
            verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        混合检索：同时进行向量检索和 BM25 关键词检索，融合结果

        Args:
            query: 查询文本（支持 OR 语法）
            index_name: 索引名称
            top_k: 返回数量
            keyword_weight: 关键词检索权重（不传则使用环境变量配置）
            vector_weight: 向量检索权重（不传则使用环境变量配置）
            similarity_threshold: 相似度阈值
            verbose: 是否打印详细信息

        Returns:
            融合后的检索结果列表
        """
        if not query:
            return []

        # 使用传入的权重或环境变量配置的权重
        kw_weight = keyword_weight if keyword_weight is not None else self.keyword_weight
        vec_weight = vector_weight if vector_weight is not None else self.vector_weight

        # 1. 向量检索
        vector_results = self._vector_search(query, index_name, top_k * 2, verbose=verbose)

        # 2. BM25 关键词检索（支持 OR 语法和同义词）
        keyword_results = self._bm25_search(query, index_name, top_k * 2, verbose=verbose)

        # 3. 融合结果
        if not vector_results and not keyword_results:
            return []
        elif not vector_results:
            results = keyword_results
        elif not keyword_results:
            results = vector_results
        else:
            results = self._fuse_results(
                vector_results, keyword_results,
                vec_weight, kw_weight
            )

        # 按最终分数排序
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        # 过滤低于阈值的结果
        filtered = [r for r in results if r.get('final_score', 0) >= similarity_threshold]

        # 打印详细结果
        if verbose and filtered:
            print("\n" + "-" * 70)
            print("混合检索结果详情:")
            print("-" * 70)
            for i, res in enumerate(filtered[:top_k], 1):
                score = res.get('final_score', 0)
                content = res.get('content_with_weight', res.get('content', ''))
                doc_name = res.get('docnm', res.get('docnm_kwd', ''))

                print(f"\n  [排名 {i}] 综合分数: {score:.4f}")
                print(f"  向量分数: {res.get('vector_score', 0):.4f}")
                print(f"  关键词分数(BM25): {res.get('keyword_score', 0):.4f}")
                print(f"  文档: {doc_name}")
                content_preview = content[:200].replace('\n', ' ')
                print(f"  内容预览: {content_preview}...")

        return filtered[:top_k]

    def _vector_search(self, query: str, index_name: str, top_k: int, verbose: bool = False) -> List[Dict]:
        """执行向量检索"""
        try:
            query_vector = self.embedding_manager.generate_embedding(query)
            if not query_vector:
                if verbose:
                    print("  向量检索: 查询向量生成失败")
                return []

            results = self.vector_store.search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k,
                similarity_threshold=0.1
            )

            if verbose and results:
                print(f"\n  向量检索召回: {len(results)} 个块")
                for i, r in enumerate(results[:3], 1):
                    score = r.get('_score', 0)
                    content = r.get('content_with_weight', r.get('content', ''))
                    print(f"    [{i}] 分数: {score:.4f} - {content[:80]}...")

            for r in results:
                r['_search_type'] = 'vector'
                r['vector_score'] = r.get('_score', 0)
                r['keyword_score'] = 0
                r['final_score'] = r.get('_score', 0)

            return results

        except Exception as e:
            if verbose:
                print(f"  向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, index_name: str, top_k: int, verbose: bool = False) -> List[Dict]:
        """执行 BM25 关键词检索 - 确保分数正确"""
        try:
            if not self.vector_store.index_exists(index_name):
                if verbose:
                    print(f"  索引不存在: {index_name}")
                return []

            doc_count = self.vector_store.get_document_count(index_name)
            if doc_count == 0:
                if verbose:
                    print("  索引为空")
                return []

            if verbose:
                print(f"  BM25检索: 索引中共有 {doc_count} 个文档")

            all_documents = self._get_all_documents(index_name)

            if not all_documents:
                if verbose:
                    print("  BM25检索: 无法获取文档")
                return []

            bm25_results = self.bm25_retriever.search(
                query=query,
                documents=all_documents,
                top_k=top_k,
                content_field="content_with_weight",
                use_or_semantics=True
            )

            if verbose and bm25_results:
                print(f"\n  BM25关键词检索召回: {len(bm25_results)} 个块")
                for i, r in enumerate(bm25_results[:3], 1):
                    # 显示归一化后的分数
                    raw_score = r.get('keyword_score_raw', 0)
                    norm_score = r.get('keyword_score', 0)
                    content = r.get('content_with_weight', r.get('content', ''))
                    print(f"    [{i}] 原始分数: {raw_score:.4f}, 归一化: {norm_score:.4f} - {content[:80]}...")

            for r in bm25_results:
                r['_search_type'] = 'bm25'
                r['vector_score'] = 0.0  # 确保是浮点数

            return bm25_results

        except Exception as e:
            if verbose:
                print(f"  BM25检索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _get_all_documents(self, index_name: str, limit: int = 2000) -> List[Dict]:
        """获取索引中的所有文档"""
        try:
            store_type = self._get_store_type()

            if store_type == 'milvus':
                return self._get_all_documents_milvus(index_name, limit)
            elif store_type == 'elasticsearch':
                return self._get_all_documents_elasticsearch(index_name, limit)
            else:
                logger.warning(f"未知的存储类型: {store_type}")
                return []

        except Exception as e:
            logger.error(f"获取所有文档失败: {e}")
            return []

    def _get_store_type(self) -> str:
        """获取当前存储类型"""
        try:
            from app.service.core.vector_store import get_store_type
            return get_store_type()
        except:
            if hasattr(self.vector_store, 'es'):
                return 'elasticsearch'
            elif hasattr(self.vector_store, '_connected'):
                return 'milvus'
            return 'unknown'

    def _get_all_documents_milvus(self, index_name: str, limit: int = 2000) -> List[Dict]:
        """从 Milvus 获取所有文档"""
        try:
            from pymilvus import Collection

            collection = Collection(index_name)
            collection.flush()
            total = min(collection.num_entities, limit)

            if total == 0:
                return []

            output_fields = [
                "id", "content", "content_with_weight", "docnm",
                "docnm_kwd", "doc_id", "kb_id", "token_count", "chunk_index"
            ]

            results = collection.query(
                expr="",
                output_fields=output_fields,
                offset=0,
                limit=total
            )

            all_docs = []
            for result in results:
                doc = {
                    "_id": result.get("id", ""),
                    "id": result.get("id", ""),
                    "content": result.get("content", ""),
                    "content_with_weight": result.get("content_with_weight", ""),
                    "docnm": result.get("docnm", ""),
                    "docnm_kwd": result.get("docnm_kwd", ""),
                    "doc_id": result.get("doc_id", ""),
                    "kb_id": result.get("kb_id", ""),
                    "token_count": result.get("token_count", 0),
                    "chunk_index": result.get("chunk_index", 0),
                }
                all_docs.append(doc)

            logger.info(f"从 Milvus 获取 {len(all_docs)} 个文档")
            return all_docs

        except Exception as e:
            logger.error(f"从 Milvus 获取文档失败: {e}")
            return []

    def _get_all_documents_elasticsearch(self, index_name: str, limit: int = 2000) -> List[Dict]:
        """从 Elasticsearch 获取所有文档"""
        try:
            body = {
                "size": limit,
                "query": {"match_all": {}},
                "_source": ["id", "content", "content_with_weight", "docnm", "docnm_kwd", "doc_id", "kb_id"]
            }

            response = self.vector_store.es.search(index=index_name, body=body)
            hits = response["hits"]["hits"]

            all_docs = []
            for hit in hits:
                doc = hit["_source"]
                doc["_id"] = hit["_id"]
                doc["id"] = doc.get("id", hit["_id"])
                all_docs.append(doc)

            logger.info(f"从 Elasticsearch 获取 {len(all_docs)} 个文档")
            return all_docs

        except Exception as e:
            logger.error(f"从 Elasticsearch 获取文档失败: {e}")
            return []

    def _fuse_results(
            self,
            vector_results: List[Dict],
            keyword_results: List[Dict],
            vector_weight: float,
            keyword_weight: float
    ) -> List[Dict]:
        """融合向量检索和 BM25 关键词检索的结果 - 修复版"""
        if not vector_results and not keyword_results:
            return []

        if not vector_results:
            return keyword_results
        if not keyword_results:
            return vector_results

        result_map = {}

        # 归一化向量分数 - 使用所有向量结果的分数范围
        all_vector_scores = [r.get('vector_score', r.get('_score', 0)) for r in vector_results if
                             r.get('_score', 0) > 0]
        if all_vector_scores:
            min_vec = min(all_vector_scores)
            max_vec = max(all_vector_scores)
            vec_range = max_vec - min_vec if max_vec > min_vec else 1.0
            logger.debug(f"向量分数范围: min={min_vec:.4f}, max={max_vec:.4f}, range={vec_range:.4f}")
        else:
            min_vec, max_vec, vec_range = 0, 1, 1

        for r in vector_results:
            doc_id = r.get('_id', r.get('id', ''))
            if not doc_id:
                continue

            raw_vec_score = r.get('vector_score', r.get('_score', 0))
            if vec_range > 0 and raw_vec_score > 0:
                # 归一化到 [0.1, 0.9] 区间
                vec_score = 0.1 + (raw_vec_score - min_vec) / vec_range * 0.8
                vec_score = max(0.05, min(0.95, vec_score))
            else:
                vec_score = 0.05

            result_map[doc_id] = {
                **r,
                'vector_score': vec_score,
                'raw_vector_score': raw_vec_score,
                'keyword_score': 0.0,
                'keyword_score_raw': 0.0,
                '_search_types': ['vector']
            }

        # 处理 BM25 结果（使用已有的归一化分数）
        # BM25 结果中的 keyword_score 已经是归一化的
        for r in keyword_results:
            doc_id = r.get('_id', r.get('id', ''))
            if not doc_id:
                continue

            # 获取 BM25 归一化分数，如果没有则计算
            kw_score = r.get('keyword_score', 0)
            if kw_score == 0 and 'keyword_score_raw' in r:
                # 如果没有归一化分数，进行简单归一化
                kw_score = min(0.95, r['keyword_score_raw'] / 10.0) if r['keyword_score_raw'] > 0 else 0.05
                kw_score = max(0.05, kw_score)

            if doc_id in result_map:
                result_map[doc_id]['keyword_score'] = kw_score
                result_map[doc_id]['keyword_score_raw'] = r.get('keyword_score_raw', 0)
                result_map[doc_id]['_search_types'].append('bm25')
            else:
                result_map[doc_id] = {
                    **r,
                    'vector_score': 0.0,
                    'keyword_score': kw_score,
                    'keyword_score_raw': r.get('keyword_score_raw', 0),
                    '_search_types': ['bm25']
                }

        # 计算融合分数（加权和）
        for doc_id, item in result_map.items():
            vector_score = item.get('vector_score', 0.0)
            keyword_score = item.get('keyword_score', 0.0)

            # 加权融合
            if vector_score > 0.05 and keyword_score > 0.05:
                item['final_score'] = (vector_score * vector_weight +
                                       keyword_score * keyword_weight)
            elif vector_score > 0.05:
                item['final_score'] = vector_score * 0.8  # 降低纯向量分数的影响
            elif keyword_score > 0.05:
                item['final_score'] = keyword_score * 0.8
            else:
                item['final_score'] = max(vector_score, keyword_score) * 0.5

            item['final_score'] = max(0.05, min(0.95, item['final_score']))

            item['fusion_info'] = {
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight,
                'vector_score': round(vector_score, 4),
                'keyword_score': round(keyword_score, 4),
                'final_score': round(item['final_score'], 4)
            }

        results = list(result_map.values())
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        return results

__all__ = ['HybridRetriever']