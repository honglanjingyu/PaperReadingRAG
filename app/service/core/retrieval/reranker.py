# app/service/core/retrieval/reranker.py
"""
重排序器
"""

import os
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_rerank_type() -> str:
    return os.getenv("RERANK_TYPE", "auto").lower()


class Reranker:
    """重排序器"""

    def __init__(self, api_type: str = None):
        self.api_type = api_type or get_rerank_type()
        self.api_key = os.getenv("RERANK_API_KEY") or os.getenv("MODEL_API_KEY")
        self.base_url = os.getenv("RERANK_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.model = os.getenv("RERANK_MODEL", "gte-rerank")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5,
               content_field: str = "content_with_weight") -> List[Dict]:
        """重排序文档"""
        if not documents or not self.api_key or not self.base_url:
            return documents[:top_k]

        texts = [doc.get(content_field, doc.get('content', '')) for doc in documents]
        texts = [t for t in texts if t and t.strip()]

        if not texts:
            return documents[:top_k]

        try:
            response = requests.post(
                self.base_url, headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": {"query": query, "documents": texts},
                      "parameters": {"top_n": min(top_k, len(texts))}},
                timeout=30
            )
            if response.status_code == 200:
                results = response.json().get("output", {}).get("results", [])
                reranked = []
                for r in results:
                    idx = r.get("index")
                    if idx is not None and idx < len(documents):
                        doc = documents[idx].copy()
                        doc['rerank_score'] = r.get("relevance_score", 0)
                        reranked.append(doc)
                reranked.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
                return reranked[:top_k]
        except Exception as e:
            logger.error(f"重排序失败: {e}")

        return documents[:top_k]


__all__ = ['Reranker', 'get_rerank_type']