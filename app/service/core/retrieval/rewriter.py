# app/service/core/retrieval/rewriter.py
"""
查询改写器
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class QueryRewriter:
    """查询改写器"""

    SYNONYMS = {
        "业务": ["业务范围", "主营业务"], "营收": ["收入", "营业收入"],
        "利润": ["净利润", "盈利"], "客户": ["客户群体", "主要客户"],
        "公司": ["企业"], "产品": ["产品线"], "市场": ["市场地位"],
    }
    STOPWORDS = {'的', '了', '是', '在', '和', '与', '或', '有', '为'}

    def rewrite(self, query: str, strategy: str = 'synonym') -> str:
        """改写查询"""
        if not query:
            return query
        if strategy == 'synonym':
            return self._expand_with_synonyms(query)
        return query

    def _expand_with_synonyms(self, query: str) -> str:
        """同义词扩展"""
        expanded = query
        for word, synonyms in self.SYNONYMS.items():
            if word in query:
                expanded = expanded.replace(word, f"({word} OR {' OR '.join(synonyms)})")
        return expanded

    def extract_keywords(self, query: str, top_n: int = 5) -> List[str]:
        """提取关键词"""
        if not query:
            return []
        keywords = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        return [kw for kw in keywords if kw not in self.STOPWORDS][:top_n]

    def add_synonyms(self, word: str, synonyms: List[str]):
        """添加自定义同义词"""
        if word in self.SYNONYMS:
            self.SYNONYMS[word].extend(synonyms)
        else:
            self.SYNONYMS[word] = synonyms


__all__ = ['QueryRewriter']