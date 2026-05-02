# app/service/core/retrieval/reranker.py
"""
重排序器 - 对检索结果进行重排序，提升召回质量
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入重排序相关模块
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers未安装，Cross-Encoder重排序不可用")


class Reranker:
    """重排序器 - 对检索结果进行精排"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        初始化重排序器

        Args:
            model_name: Cross-Encoder模型名称
        """
        self.cross_encoder = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(model_name)
                logger.info(f"Cross-Encoder模型加载成功: {model_name}")
            except Exception as e:
                logger.warning(f"Cross-Encoder模型加载失败: {e}")
        else:
            logger.info("使用向量相似度重排序（替代Cross-Encoder）")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        content_field: str = "content_with_weight"
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 检索结果列表
            top_k: 返回数量
            content_field: 内容字段名

        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []

        # 提取文档内容
        contents = [doc.get(content_field, doc.get('content', '')) for doc in documents]

        # 使用Cross-Encoder重排序（如果有）
        if self.cross_encoder and CROSS_ENCODER_AVAILABLE:
            return self._rerank_with_cross_encoder(query, documents, contents, top_k)
        else:
            return self._rerank_with_vector_similarity(query, documents, contents, top_k)

    def _rerank_with_cross_encoder(
        self,
        query: str,
        documents: List[Dict],
        contents: List[str],
        top_k: int
    ) -> List[Dict]:
        """使用Cross-Encoder重排序"""
        try:
            # 构建(query, document)对
            pairs = [(query, content) for content in contents]

            # 计算相似度分数
            scores = self.cross_encoder.predict(pairs)

            # 添加分数并排序
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
                doc['original_score'] = doc.get('final_score', doc.get('_score', 0))

            # 按重排序分数排序
            documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

            logger.info(f"Cross-Encoder重排序完成: {len(documents)} -> {top_k}")
            return documents[:top_k]

        except Exception as e:
            logger.error(f"Cross-Encoder重排序失败: {e}")
            return self._rerank_with_vector_similarity(query, documents, contents, top_k)

    def _rerank_with_vector_similarity(
        self,
        query: str,
        documents: List[Dict],
        contents: List[str],
        top_k: int
    ) -> List[Dict]:
        """使用向量相似度重排序（备用方案）"""
        try:
            from app.service.core.embedding import get_embedding_manager

            embedding_manager = get_embedding_manager()
            query_vector = embedding_manager.generate_embedding(query)

            if not query_vector:
                return documents[:top_k]

            # 计算每个文档的向量相似度
            for doc in documents:
                # 尝试从文档中获取向量
                doc_vector = None
                for key, value in doc.items():
                    if key.endswith('_vec') and isinstance(value, list):
                        doc_vector = value
                        break

                if doc_vector and len(doc_vector) == len(query_vector):
                    # 计算余弦相似度
                    sim = np.dot(query_vector, doc_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-8
                    )
                    doc['rerank_score'] = float(sim)
                else:
                    # 如果没有向量，保留原始分数
                    doc['rerank_score'] = doc.get('final_score', doc.get('_score', 0))

                doc['original_score'] = doc.get('final_score', doc.get('_score', 0))

            # 按重排序分数排序
            documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

            logger.info(f"向量相似度重排序完成: {len(documents)} -> {top_k}")
            return documents[:top_k]

        except Exception as e:
            logger.error(f"向量相似度重排序失败: {e}")
            return documents[:top_k]

    def rerank_with_llm(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        llm_client=None
    ) -> List[Dict]:
        """
        使用LLM进行重排序（可选，需要LLM客户端）

        Args:
            query: 查询文本
            documents: 检索结果
            top_k: 返回数量
            llm_client: LLM客户端（如DashScope）

        Returns:
            重排序后的结果
        """
        if not llm_client or len(documents) <= top_k:
            return documents[:top_k]

        try:
            # 构建LLM重排序提示
            prompt = self._build_rerank_prompt(query, documents)

            # 调用LLM获取排序结果
            response = llm_client.chat.completions.create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            # 解析LLM返回的排序结果
            ranked_indices = self._parse_llm_ranking(response.choices[0].message.content)

            # 按LLM排序结果重组
            if ranked_indices:
                reranked = [documents[i] for i in ranked_indices if i < len(documents)]
                return reranked[:top_k]

        except Exception as e:
            logger.error(f"LLM重排序失败: {e}")

        return documents[:top_k]

    def _build_rerank_prompt(self, query: str, documents: List[Dict]) -> str:
        """构建LLM重排序提示"""
        prompt = f"请根据以下查询，对提供的文档片段进行相关性排序（从最相关到最相关）。\n\n"
        prompt += f"查询: {query}\n\n"
        prompt += "文档列表:\n"

        for i, doc in enumerate(documents):
            content = doc.get('content_with_weight', doc.get('content', ''))[:200]
            prompt += f"{i+1}. {content}...\n\n"

        prompt += "请只返回文档编号的排序结果，格式如: 3,1,2,5,4"
        return prompt

    def _parse_llm_ranking(self, llm_response: str) -> List[int]:
        """解析LLM返回的排序结果"""
        try:
            import re
            numbers = re.findall(r'\d+', llm_response)
            return [int(n) - 1 for n in numbers]  # 转换为0索引
        except Exception:
            return []


__all__ = ['Reranker']