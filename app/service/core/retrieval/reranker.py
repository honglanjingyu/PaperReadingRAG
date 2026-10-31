# app/service/core/retrieval/reranker.py
"""
重排序器
"""

import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器（远程 API）"""

    def __init__(self, api_type: str = None):
        self.api_type = "remote"
        self.api_key = os.getenv("RERANK_API_KEY") or os.getenv("MODEL_API_KEY")
        self.base_url = os.getenv("RERANK_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.model = os.getenv("RERANK_MODEL", "gte-rerank")
        # 最近一次 rerank 的可观测信息（供上层透传到响应/评测报告）
        self.last_meta: Dict[str, Any] = {
            "rerank_applied": False,
            "rerank_reason": "not_called",
            "rerank_model": self.model,
        }

    def _set_meta(self, applied: bool, reason: str, **extra) -> None:
        """记录最近一次 rerank 的执行结果，避免静默降级。"""
        self.last_meta = {
            "rerank_applied": applied,
            "rerank_reason": reason,
            "rerank_model": self.model,
        }
        self.last_meta.update(extra)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5,
               content_field: str = "content_with_weight") -> List[Dict]:
        """重排序文档

        每次调用都会在 self.last_meta 里记录是否真正执行、未执行的原因；
        失败一律降级为原序，但不再静默。
        """
        total = len(documents or [])

        if not documents:
            self._set_meta(False, "empty_documents", input_count=0, output_count=0)
            logger.warning("rerank 跳过: empty_documents（无候选文档）")
            return documents[:top_k]

        if not self.api_key:
            self._set_meta(False, "no_key", input_count=total, output_count=min(top_k, total))
            logger.warning("rerank 跳过: no_key（RERANK_API_KEY / MODEL_API_KEY 未配置）")
            return documents[:top_k]

        if not self.base_url:
            self._set_meta(False, "no_base_url", input_count=total, output_count=min(top_k, total))
            logger.warning("rerank 跳过: no_base_url（RERANK_BASE_URL / LLM_BASE_URL 未配置）")
            return documents[:top_k]

        # 保留原下标：远程 rerank 返回的 index 是「请求里第几条」的序号，
        # 一旦过滤掉空文本，序号就会与 documents 错位，必须映射回原下标。
        indexed_texts = [
            (i, doc.get(content_field, doc.get('content', '')))
            for i, doc in enumerate(documents)
        ]
        indexed_texts = [(i, t) for i, t in indexed_texts if t and t.strip()]
        texts = [t for _, t in indexed_texts]

        if not texts:
            self._set_meta(False, "empty_texts", input_count=total, output_count=min(top_k, total))
            logger.warning(f"rerank 跳过: empty_texts（{total} 条候选内容全为空）")
            return documents[:top_k]

        t0 = time.perf_counter()
        try:
            response = requests.post(
                self.base_url, headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": {"query": query, "documents": texts},
                      "parameters": {"top_n": min(top_k, len(texts))}},
                timeout=30
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if response.status_code == 200:
                results = response.json().get("output", {}).get("results", [])
                reranked = []
                for r in results:
                    sent_idx = r.get("index")
                    # sent_idx 是「过滤后发送列表」的下标，要映射回 documents 的原下标
                    if sent_idx is None or sent_idx < 0 or sent_idx >= len(indexed_texts):
                        continue
                    orig_idx = indexed_texts[sent_idx][0]
                    doc = documents[orig_idx].copy()
                    doc['rerank_score'] = r.get("relevance_score", 0)
                    doc['original_rank'] = orig_idx + 1
                    reranked.append(doc)
                reranked.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
                kept = reranked[:top_k]
                for new_rank, doc in enumerate(kept, 1):
                    doc['new_rank'] = new_rank
                rank_changes = [
                    {
                        'parent_id': d.get('parent_id', ''),
                        'original_rank': d.get('original_rank'),
                        'new_rank': d.get('new_rank'),
                        'delta': (d.get('original_rank') or 0) - (d.get('new_rank') or 0),
                    }
                    for d in kept
                ]
                moved = sum(1 for c in rank_changes if c['delta'] != 0)
                self._set_meta(
                    True, "applied",
                    input_count=total,
                    sent_count=len(texts),
                    output_count=len(kept),
                    rerank_ms=elapsed_ms,
                    rank_changed_count=moved,
                    rank_changes=rank_changes,
                )
                logger.info(
                    f"rerank 生效: model={self.model} 候选={total} 请求={len(texts)} "
                    f"返回={len(kept)} 名次变动={moved} 耗时={elapsed_ms}ms"
                )
                return kept

            self._set_meta(
                False, f"http_{response.status_code}",
                input_count=total, sent_count=len(texts),
                output_count=min(top_k, total),
            )
            logger.warning(
                f"rerank 失败: http_{response.status_code}，已降级为原序（候选 {total} 条）"
            )
        except requests.exceptions.Timeout as e:
            cost_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            self._set_meta(
                False, "timeout", input_count=total, sent_count=len(texts),
                output_count=min(top_k, total), rerank_ms=cost_ms,
            )
            logger.warning(
                f"rerank 失败: timeout，已降级为原序（候选 {total} 条，耗时 {cost_ms}ms）: {e}"
            )
        except Exception as e:
            cost_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            self._set_meta(
                False, f"error_{type(e).__name__}",
                input_count=total, sent_count=len(texts),
                output_count=min(top_k, total), rerank_ms=cost_ms,
            )
            logger.error(
                f"rerank 失败: {type(e).__name__}，已降级为原序（耗时 {cost_ms}ms）: {e}"
            )

        return documents[:top_k]


__all__ = ['Reranker']