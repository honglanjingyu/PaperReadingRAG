# app/service/core/retrieval/hybrid_retriever.py

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器 - 融合关键词检索和向量检索"""

    def __init__(self, vector_store=None, embedding_manager=None):
        """
        初始化混合检索器

        Args:
            vector_store: 向量存储实例（ESVectorStore 或 MilvusVectorStore）
            embedding_manager: EmbeddingManager 实例
        """
        from app.service.core.vector_store import get_vector_store
        from app.service.core.embedding import get_embedding_manager

        # 使用工厂模式获取正确的向量存储实例
        self.vector_store = vector_store or get_vector_store()
        self.embedding_manager = embedding_manager or get_embedding_manager()
        self.keyword_weight = 0.3
        self.vector_weight = 0.7

    def hybrid_search(
            self,
            query: str,
            index_name: str,
            top_k: int = 5,
            keyword_weight: float = 0.3,
            vector_weight: float = 0.7,
            similarity_threshold: float = 0.3,
            verbose: bool = False  # 新增 verbose 参数
    ) -> List[Dict[str, Any]]:
        """
        混合检索：同时进行关键词检索和向量检索，融合结果

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回数量
            keyword_weight: 关键词检索权重
            vector_weight: 向量检索权重
            similarity_threshold: 相似度阈值
            verbose: 是否打印详细信息

        Returns:
            融合后的检索结果列表
        """
        if not query:
            return []

        # 1. 向量检索
        vector_results = self._vector_search(query, index_name, top_k * 2, verbose=verbose)

        # 2. 关键词检索（仅 Elasticsearch 支持，Milvus 暂不支持）
        keyword_results = self._keyword_search(query, index_name, top_k * 2, verbose=verbose)

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
                vector_weight, keyword_weight
            )

        # 关键修复：按最终分数排序
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
                print(f"  关键词分数: {res.get('keyword_score', 0):.4f}")
                print(f"  文档: {doc_name}")
                content_preview = content[:200].replace('\n', ' ')
                print(f"  内容预览: {content_preview}...")

        return filtered[:top_k]

    def _vector_search(self, query: str, index_name: str, top_k: int, verbose: bool = False) -> List[Dict]:
        """执行向量检索"""
        try:
            # 生成查询向量
            query_vector = self.embedding_manager.generate_embedding(query)
            if not query_vector:
                if verbose:
                    print("  向量检索: 查询向量生成失败")
                return []

            # 执行向量搜索（使用统一的 vector_store 接口）
            results = self.vector_store.search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k,
                similarity_threshold=0.1
            )

            if verbose and results:
                print(f"\n  向量检索召回: {len(results)} 个块")
                # 打印向量检索结果预览
                for i, r in enumerate(results[:3], 1):
                    score = r.get('_score', 0)
                    content = r.get('content_with_weight', r.get('content', ''))
                    print(f"    [{i}] 分数: {score:.4f} - {content[:80]}...")

            # 添加结果类型标记
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

    def _keyword_search(self, query: str, index_name: str, top_k: int, verbose: bool = False) -> List[Dict]:
        """执行关键词检索（仅 Elasticsearch 支持）"""
        try:
            # 检查是否为 Elasticsearch 实例（有 es 属性）
            if not hasattr(self.vector_store, 'es') or self.vector_store.es is None:
                # Milvus 不支持关键词检索，直接返回空
                if verbose:
                    print("  关键词检索: Milvus 不支持，跳过")
                return []

            if not self.vector_store.es.indices.exists(index=index_name):
                if verbose:
                    print(f"  关键词检索: 索引 {index_name} 不存在")
                return []

            # 构建关键词查询
            body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^3", "content_with_weight^2", "docnm^1.5"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                }
            }

            response = self.vector_store.es.search(index=index_name, body=body)

            hits = response["hits"]["hits"]
            if not hits:
                if verbose:
                    print("  关键词检索: 无结果")
                return []

            # 收集所有原始分数
            raw_scores = [hit.get("_score", 0) for hit in hits]
            min_score = min(raw_scores)
            max_score = max(raw_scores)
            score_range = max_score - min_score if max_score > min_score else 1.0

            if score_range < 1.0:
                score_range = 1.0

            results = []
            for hit in hits:
                raw_score = hit.get("_score", 0)

                if score_range > 0:
                    norm_score = (raw_score - min_score) / score_range
                else:
                    norm_score = 0.5

                norm_score = max(0.05, min(0.95, norm_score))

                result = hit["_source"]
                result["_id"] = hit["_id"]
                result["_search_type"] = "keyword"
                result["keyword_score"] = norm_score
                result["raw_keyword_score"] = raw_score
                result["vector_score"] = 0
                result["final_score"] = norm_score
                results.append(result)

            results.sort(key=lambda x: x["raw_keyword_score"], reverse=True)

            if verbose:
                print(f"\n  关键词检索召回: {len(results)} 个块")
                for i, r in enumerate(results[:3], 1):
                    score = r.get('raw_keyword_score', 0)
                    content = r.get('content_with_weight', r.get('content', ''))
                    print(f"    [{i}] 分数: {score:.4f} - {content[:80]}...")

            return results

        except Exception as e:
            if verbose:
                print(f"  关键词检索失败: {e}")
            return []

    def _fuse_results(
            self,
            vector_results: List[Dict],
            keyword_results: List[Dict],
            vector_weight: float,
            keyword_weight: float
    ) -> List[Dict]:
        """融合向量检索和关键词检索的结果"""
        if not vector_results and not keyword_results:
            return []

        # 如果只有一种结果，直接返回
        if not vector_results:
            return keyword_results
        if not keyword_results:
            return vector_results

        result_map = {}

        # 处理向量检索结果 - 先归一化分数
        vec_raw_scores = [r.get('_score', 0) for r in vector_results]
        if vec_raw_scores:
            min_vec = min(vec_raw_scores)
            max_vec = max(vec_raw_scores)
            vec_range = max_vec - min_vec if max_vec > min_vec else 1.0
        else:
            min_vec, max_vec, vec_range = 0, 1, 1

        for r in vector_results:
            doc_id = r.get('_id', r.get('id', ''))
            if not doc_id:
                continue

            raw_vec_score = r.get('_score', 0)
            if vec_range > 0:
                vec_score = max(0, min(1, (raw_vec_score - min_vec) / vec_range))
            else:
                vec_score = 0.5

            result_map[doc_id] = {
                **r,
                'vector_score': vec_score,
                'raw_vector_score': raw_vec_score,
                'keyword_score': 0,
                'raw_keyword_score': 0,
                '_search_types': ['vector']
            }

        # 处理关键词检索结果 - 先归一化分数
        kw_raw_scores = [r.get('raw_keyword_score', r.get('keyword_score', 0)) for r in keyword_results]
        if kw_raw_scores:
            min_kw = min(kw_raw_scores)
            max_kw = max(kw_raw_scores)
            kw_range = max_kw - min_kw if max_kw > min_kw else 1.0
        else:
            min_kw, max_kw, kw_range = 0, 1, 1

        for r in keyword_results:
            doc_id = r.get('_id', r.get('id', ''))
            if not doc_id:
                continue

            raw_kw_score = r.get('raw_keyword_score', r.get('keyword_score', 0))

            if kw_range > 0:
                kw_score = max(0, min(1, (raw_kw_score - min_kw) / kw_range))
            else:
                kw_score = 0.5

            if doc_id in result_map:
                result_map[doc_id]['keyword_score'] = kw_score
                result_map[doc_id]['raw_keyword_score'] = raw_kw_score
                result_map[doc_id]['_search_types'].append('keyword')
            else:
                result_map[doc_id] = {
                    **r,
                    'vector_score': 0,
                    'keyword_score': kw_score,
                    'raw_keyword_score': raw_kw_score,
                    '_search_types': ['keyword']
                }

        # 计算融合分数
        for doc_id, item in result_map.items():
            vector_score = item.get('vector_score', 0)
            keyword_score = item.get('keyword_score', 0)

            vector_score = max(0, min(1, vector_score))
            keyword_score = max(0, min(1, keyword_score))

            if vector_score > 0.05 and keyword_score > 0.05:
                item['final_score'] = (vector_score * vector_weight +
                                       keyword_score * keyword_weight)
            elif vector_score > 0.05:
                item['final_score'] = vector_score
            elif keyword_score > 0.05:
                item['final_score'] = keyword_score
            else:
                item['final_score'] = max(vector_score, keyword_score)

            item['final_score'] = max(0, min(1, item['final_score']))

            item['fusion_info'] = {
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight,
                'vector_score': round(vector_score, 4),
                'keyword_score': round(keyword_score, 4)
            }

        results = list(result_map.values())

        # 关键修复：按 final_score 降序排序
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        return results


__all__ = ['HybridRetriever']