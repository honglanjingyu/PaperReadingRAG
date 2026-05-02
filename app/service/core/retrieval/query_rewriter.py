# app/service/core/retrieval/query_rewriter.py
"""
Query改写器 - 同义词扩展、关键词提取、子查询生成
"""

import re
import logging
from typing import List, Set, Dict, Optional

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Query改写器 - 提升查询质量"""

    # 同义词词典（可扩展）
    SYNONYMS = {
        # 业务相关
        "业务": ["业务范围", "主营业务", "经营范围"],
        "营收": ["收入", "营业收入", "销售额"],
        "利润": ["净利润", "盈利", "收益"],
        "客户": ["客户群体", "主要客户", "客户结构"],
        "竞争优势": ["核心竞争力", "优势", "竞争壁垒"],
        "发展": ["发展态势", "发展趋势", "发展前景"],
        "风险": ["风险因素", "经营风险", "潜在风险"],

        # 财报相关
        "中报": ["半年报", "中期报告", "半年度报告"],
        "年报": ["年度报告", "年报披露"],
        "营收增长": ["收入增长", "收入增速", "营收增速"],
        "毛利率": ["毛利水平", "毛利率水平"],

        # 通用
        "公司": ["企业", "上市公司"],
        "产品": ["产品线", "产品结构"],
        "市场": ["市场地位", "市场份额"],
    }

    def __init__(self, synonym_dict: Dict[str, List[str]] = None):
        """
        初始化Query改写器

        Args:
            synonym_dict: 自定义同义词词典
        """
        self.synonym_dict = synonym_dict or self.SYNONYMS.copy()

    def rewrite(self, query: str, strategy: str = 'synonym') -> str:
        """
        改写查询

        Args:
            query: 原始查询
            strategy: 改写策略 ('synonym', 'expand', 'simplify')

        Returns:
            改写后的查询
        """
        if not query:
            return query

        if strategy == 'synonym':
            return self._expand_with_synonyms(query)
        elif strategy == 'expand':
            return self._expand_query(query)
        elif strategy == 'simplify':
            return self._simplify_query(query)
        else:
            return query

    def _expand_with_synonyms(self, query: str) -> str:
        """使用同义词扩展查询"""
        expanded_query = query
        for word, synonyms in self.synonym_dict.items():
            if word in query:
                # 添加同义词，使用 OR 连接
                synonym_part = " OR ".join(synonyms)
                expanded_query = expanded_query.replace(word, f"({word} OR {synonym_part})")
        return expanded_query

    def extract_keywords(self, query: str, top_n: int = 5) -> List[str]:
        """
        提取查询中的关键词

        Args:
            query: 查询文本
            top_n: 返回的关键词数量

        Returns:
            关键词列表
        """
        if not query:
            return []

        # 简单的中文分词（按字符和常见词分割）
        # 可以使用 rag_tokenizer 进行更精确的分词
        try:
            from app.service.core.rag.nlp import rag_tokenizer
            tokens = rag_tokenizer.tokenize(query).split()
            return tokens[:top_n]
        except ImportError:
            # 简单的正则提取
            keywords = re.findall(r'[\u4e00-\u9fff]{2,}', query)
            # 移除常见的停用词
            stopwords = {'的', '了', '是', '在', '和', '与', '或', '有', '为', '对', '从', '到'}
            keywords = [kw for kw in keywords if kw not in stopwords]
            return keywords[:top_n]

    def generate_sub_queries(self, query: str, max_queries: int = 3) -> List[str]:
        """
        生成子查询，用于搜索时获得更多样化的结果

        Args:
            query: 原始查询
            max_queries: 最大子查询数量

        Returns:
            子查询列表
        """
        sub_queries = [query]  # 原始查询

        keywords = self.extract_keywords(query, top_n=3)

        # 基于关键词生成子查询
        if len(keywords) > 1:
            sub_queries.append(" ".join(keywords[:2]))
        if len(keywords) > 2:
            sub_queries.append(keywords[0])

        # 去重
        sub_queries = list(dict.fromkeys(sub_queries))

        return sub_queries[:max_queries]

    def _expand_query(self, query: str) -> str:
        """扩展查询（添加更多相关词）"""
        keywords = self.extract_keywords(query)
        if not keywords:
            return query

        # 为每个关键词添加同义词
        expanded_parts = []
        for kw in keywords:
            synonyms = []
            for word, syn_list in self.synonym_dict.items():
                if kw in word or word in kw:
                    synonyms.extend(syn_list)

            if synonyms:
                expanded_parts.append(f"{kw} OR {' OR '.join(synonyms[:2])}")
            else:
                expanded_parts.append(kw)

        return " ".join(expanded_parts)

    def _simplify_query(self, query: str) -> str:
        """简化查询（保留核心词）"""
        keywords = self.extract_keywords(query, top_n=3)
        return " ".join(keywords) if keywords else query

    def add_synonyms(self, word: str, synonyms: List[str]):
        """添加自定义同义词"""
        if word in self.synonym_dict:
            self.synonym_dict[word].extend(synonyms)
        else:
            self.synonym_dict[word] = synonyms


__all__ = ['QueryRewriter']