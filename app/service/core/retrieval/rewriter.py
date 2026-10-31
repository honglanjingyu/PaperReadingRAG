# app/service/core/retrieval/rewriter.py
"""
查询改写器 —— 术语归一（Q1 + Q2）

设计要点（见 work/eval/reports/reranker与query改写优化方案.md 第 3 节）：

Q1：原 `SYNONYMS` 业务同义词表（营收/利润/客户…）对论文语料零命中，且产出的
    `(A OR B)` 字面串会同时污染 ES 的 multi_match 和 embedding 向量路，现停用。
    本类不再产出任何「拼好的字符串」，只返回**术语扩展词列表**，由调用方决定
    只用在哪一路（当前只在 BM25 路，向量路永远用原 query）。

Q2：改为面向论文语料的**领域术语归一**。词表配置化放在同目录 `term_map.json`，
    可用环境变量 `TERM_MAP_PATH` 指向自定义 JSON 覆盖；不在 .py 里硬编码术语。
    命中规则：query 中出现某条的 canonical 或任一 alias（大小写不敏感、子串匹配），
    就把该条的全部 expansions 收集为扩展词，交给 ES 追加成 should 子句。
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# 默认词表就放在本模块同目录，随代码一起走
_DEFAULT_TERM_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "term_map.json")


class QueryRewriter:
    """查询改写器（术语归一版）

    对外只暴露两件事：
    - `expand_terms(query)` -> List[str]：命中的扩展词，给 BM25 用；
    - `rewrite(query, strategy=...)` -> str：仅为兼容旧调用点保留，
      strategy='term' 时返回「原 query + 扩展词」的可读字符串（**不要拿去 embedding**），
      strategy='none' 或默认直接返回原 query。
    """

    STOPWORDS = {'的', '了', '是', '在', '和', '与', '或', '有', '为'}

    def __init__(self, term_map_path: str = None):
        self.term_map_path = term_map_path or os.getenv("TERM_MAP_PATH") or _DEFAULT_TERM_MAP
        self._terms: List[Dict] = []
        self._loaded = False
        self.load_terms()

    # ---------- 词表加载 ----------

    def load_terms(self) -> None:
        """加载术语归一表；文件缺失或格式不对时静默降级为空表（不影响检索主链路）。"""
        self._loaded = False
        self._terms = []
        try:
            with open(self.term_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_terms = data.get("terms") or []
            terms = []
            for item in raw_terms:
                canonical = (item.get("canonical") or "").strip()
                if not canonical:
                    continue
                aliases = [a.strip() for a in (item.get("aliases") or []) if a and a.strip()]
                expansions = [e.strip() for e in (item.get("expansions") or []) if e and e.strip()]
                # 触发词 = canonical + aliases，统一小写做匹配
                triggers = {canonical.lower()} | {a.lower() for a in aliases}
                terms.append({
                    "canonical": canonical,
                    "triggers": sorted(triggers, key=len, reverse=True),
                    "expansions": expansions,
                })
            self._terms = terms
            self._loaded = True
            logger.info(f"术语归一表已加载: {self.term_map_path}，共 {len(terms)} 条")
        except FileNotFoundError:
            logger.warning(f"术语归一表不存在，术语扩展关闭: {self.term_map_path}")
        except Exception as e:
            logger.warning(f"术语归一表加载失败，术语扩展关闭: {type(e).__name__}: {e}")

    @property
    def is_ready(self) -> bool:
        return self._loaded and bool(self._terms)

    # ---------- 术语归一 ----------

    @staticmethod
    def _trigger_hit(trigger: str, q: str) -> bool:
        """判断某个触发词是否命中 query（q 已小写）。

        纯 ASCII 触发词（PE / QKV / MHA / attention 等）按**词边界**匹配，
        避免短词误命中：PE 不再命中 pipeline、paper、type 里的 pe。
        含中文的触发词仍用子串匹配 —— 中文没有空格分词，子串即是词。
        """
        if not trigger:
            return False
        if re.fullmatch(r"[\x00-\x7f]+", trigger):
            pattern = r"(?<![a-z0-9])" + re.escape(trigger) + r"(?![a-z0-9])"
            return re.search(pattern, q) is not None
        return trigger in q

    def expand_terms(self, query: str) -> List[str]:
        """识别 query 中的领域术语，返回要追加给 BM25 的扩展词（去重、保序）。

        只返回 expansions，不返回原 query —— 原 query 由 ES 的 must 子句负责。
        """
        if not query or not self.is_ready:
            return []
        q = query.lower()
        seen = set()
        expansions: List[str] = []
        for term in self._terms:
            hit = any(self._trigger_hit(t, q) for t in term["triggers"])
            if not hit:
                continue
            for e in term["expansions"]:
                key = e.lower()
                if key in seen:
                    continue
                seen.add(key)
                expansions.append(e)
        return expansions

    # ---------- 兼容旧调用点 ----------

    def rewrite(self, query: str, strategy: str = 'none') -> str:
        """改写查询。

        Q1 之后默认不再改写：strategy 默认 'none' 直接返回原 query。
        传 'term' 时返回「原 query + 扩展词」的字符串，仅供调试/日志观察，
        绝不能把它拿去生成 embedding（会在向量路引入噪声）。
        """
        if not query:
            return query
        if strategy == 'term':
            expansions = self.expand_terms(query)
            if expansions:
                return query + " " + " ".join(expansions)
        return query

    def _expand_with_synonyms(self, query: str) -> str:
        """已停用（Q1）。保留空实现只为兼容外部可能存在的调用，不再产出 (A OR B)。"""
        return query

    def extract_keywords(self, query: str, top_n: int = 5) -> List[str]:
        """提取关键词"""
        if not query:
            return []
        keywords = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        return [kw for kw in keywords if kw not in self.STOPWORDS][:top_n]

    def add_synonyms(self, word: str, synonyms: List[str]):
        """已停用（Q1）：业务同义词表不再维护。保留空实现避免旧调用报错。"""
        logger.debug("QueryRewriter.add_synonyms 已停用，术语请改到 term_map.json")


__all__ = ['QueryRewriter']
