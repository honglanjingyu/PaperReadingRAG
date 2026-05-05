# app/service/core/retrieval/bm25_retriever.py
"""
BM25 关键词检索器 - 支持 OR 语法和同义词扩展
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

try:
    from rank_bm25 import BM25Okapi, BM25Plus, BM25L

    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    logging.warning("rank-bm25 未安装，BM25 检索不可用")

try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("jieba 未安装，使用简化分词")

# 导入同义词加载器
from app.service.synonymlist.loader import get_synonym_loader

logger = logging.getLogger(__name__)


class BM25Variant(Enum):
    """BM25 变体类型"""
    OKAPI = "okapi"
    PLUS = "plus"
    L = "l"


class BM25Retriever:
    """
    BM25 关键词检索器 - 支持 OR 语法

    功能：
    1. 解析 Query Rewriter 生成的 OR 表达式
    2. 支持从 YAML 文件加载同义词表
    3. 支持多种 BM25 变体
    """

    def __init__(
            self,
            variant: BM25Variant = BM25Variant.OKAPI,
            use_jieba: bool = True,
            use_synonyms: bool = True,
            k1: float = 1.5,
            b: float = 0.75,
            delta: float = 1.0
    ):
        """
        初始化 BM25 检索器

        Args:
            variant: BM25 变体类型
            use_jieba: 是否使用 jieba 分词
            use_synonyms: 是否使用同义词扩展
            k1: BM25 参数
            b: BM25 参数
            delta: BM25Plus 参数
        """
        if not RANK_BM25_AVAILABLE:
            raise ImportError("请安装 rank-bm25: pip install rank-bm25==0.2.2")

        self.variant = variant
        self.use_jieba = use_jieba and JIEBA_AVAILABLE
        self.use_synonyms = use_synonyms
        self.k1 = k1
        self.b = b
        self.delta = delta

        self._bm25_model = None
        self._corpus = []
        self._tokenized_corpus = []

        # 加载同义词表
        self._synonyms = {}
        if use_synonyms:
            try:
                loader = get_synonym_loader()
                self._synonyms = loader.get_all_synonyms()
                logger.info(f"BM25Retriever 加载同义词: {len(self._synonyms)} 个词条")
            except Exception as e:
                logger.warning(f"加载同义词失败: {e}")

        if self.use_jieba:
            logger.info("BM25Retriever 已启用 jieba 分词")
        else:
            logger.info("BM25Retriever 使用简化分词")

    def _tokenize(self, text: str) -> List[str]:
        """分词处理"""
        if not text:
            return []

        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fff\u0041-\u005a\u0061-\u007a0-9\s]', ' ', text)

        if self.use_jieba:
            words = jieba.lcut(text)
            words = [w for w in words if len(w) > 1]
        else:
            words = []
            chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
            words.extend(chinese_words)
            english_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
            words.extend(english_words)
            numbers = re.findall(r'\d+', text)
            words.extend(numbers)

        return list(set(words))

    def parse_or_query(self, query: str) -> List[str]:
        """
        解析 OR 查询，返回搜索词列表

        示例：
            输入: "世运电子的主要(业务 OR 业务范围 OR 主营业务)是什么？"
            输出: ['世运', '电子', '主要', '业务', '业务范围', '主营业务', '是什么']

        Args:
            query: 包含 OR 语法的查询字符串

        Returns:
            搜索词列表
        """
        if not query:
            return []

        all_terms = []

        # 匹配 OR 表达式: (词1 OR 词2 OR 词3)
        or_pattern = r'\(([^)]+)\)'

        # 找到所有 OR 组并提取词
        for match in re.findall(or_pattern, query):
            terms = [t.strip() for t in match.split(' OR ')]
            all_terms.extend(terms)
            # 从查询中移除已处理的 OR 组（用空替换）
            query = query.replace(f"({match})", "")

        # 处理剩余文本
        remaining_terms = self._tokenize(query)
        all_terms.extend(remaining_terms)

        # 去重
        return list(set(all_terms))

    def expand_with_synonyms(self, terms: List[str]) -> List[str]:
        """
        使用同义词扩展词列表

        Args:
            terms: 原始词列表

        Returns:
            扩展后的词列表（包含原始词和同义词）
        """
        if not self.use_synonyms:
            return terms

        expanded = []
        for term in terms:
            expanded.append(term)
            # 查找同义词
            for word, syns in self._synonyms.items():
                # 精确匹配
                if term == word:
                    expanded.extend(syns)
                # 词包含关系
                elif term in word or word in term:
                    expanded.extend(syns)
                # 检查同义词匹配
                for syn in syns:
                    if syn == term or (term in syn or syn in term):
                        expanded.append(word)
                        expanded.extend(syns)
                        break

        return list(set(expanded))

    def build_corpus(self, documents: List[Dict], content_field: str = "content_with_weight") -> int:
        """
        构建 BM25 语料库

        Args:
            documents: 文档列表
            content_field: 内容字段名

        Returns:
            语料库大小
        """
        if not documents:
            return 0

        self._corpus = []
        self._tokenized_corpus = []

        for doc in documents:
            content = doc.get(content_field, doc.get('content', ''))
            if content and content.strip():
                tokens = self._tokenize(content)
                if tokens:
                    self._tokenized_corpus.append(tokens)
                    self._corpus.append(doc)

        if not self._tokenized_corpus:
            logger.warning("没有有效的语料数据")
            return 0

        # 创建 BM25 模型
        if self.variant == BM25Variant.PLUS:
            self._bm25_model = BM25Plus(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
                delta=self.delta
            )
        elif self.variant == BM25Variant.L:
            self._bm25_model = BM25L(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b
            )
        else:
            self._bm25_model = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b
            )

        logger.info(f"BM25 语料库构建完成: {len(self._corpus)} 个文档")
        return len(self._corpus)

    def search(
            self,
            query: str,
            documents: List[Dict] = None,
            top_k: int = 5,
            content_field: str = "content_with_weight",
            use_or_semantics: bool = True,
            synonym_weight: float = 0.8
    ) -> List[Dict]:
        """使用 BM25 进行检索 - 修复版"""
        if not query:
            return []

        if documents is not None:
            self.build_corpus(documents, content_field)

        if not self._bm25_model or not self._corpus:
            logger.warning("BM25 模型未初始化")
            return []

        # 解析 OR 查询
        search_terms = self.parse_or_query(query)

        if not search_terms:
            logger.warning("查询解析后为空")
            return []

        # 同义词扩展
        expanded_terms = self.expand_with_synonyms(search_terms)

        logger.debug(f"搜索词: {search_terms}")
        logger.debug(f"扩展后: {expanded_terms}")

        if use_or_semantics:
            doc_scores = self._search_with_or_semantics(expanded_terms)
        else:
            doc_scores = self._search_with_weighted_semantics(search_terms, expanded_terms, synonym_weight)

        # 找到最大分数用于归一化
        max_score = max(doc_scores) if doc_scores else 1.0

        # 构建结果
        results = []
        for idx, score in enumerate(doc_scores):
            if score > 0 and self._corpus[idx].get('content_with_weight'):
                result = self._corpus[idx].copy()
                result['keyword_score_raw'] = score
                # 归一化：使用 score / max_score，缩放到 0.1-0.9
                if max_score > 0:
                    norm_score = 0.1 + (score / max_score) * 0.8
                else:
                    norm_score = 0.5
                result['keyword_score'] = max(0.05, min(0.95, norm_score))
                result['_search_type'] = 'bm25'
                result['_bm25_variant'] = self.variant.value
                results.append(result)

        # 按原始分数排序
        results.sort(key=lambda x: x.get('keyword_score_raw', 0), reverse=True)

        logger.info(f"BM25 检索完成: 返回 {len(results[:top_k])} 个结果")
        return results[:top_k]

    def _search_with_or_semantics(self, terms: List[str]) -> List[float]:
        """
        OR 语义搜索：对每个词取最大值

        Args:
            terms: 搜索词列表

        Returns:
            每个文档的分数列表
        """
        doc_count = len(self._corpus)
        doc_scores = [0.0] * doc_count

        for term in terms:
            scores = self._bm25_model.get_scores([term])
            for i in range(doc_count):
                doc_scores[i] = max(doc_scores[i], scores[i])

        return doc_scores

    def _search_with_weighted_semantics(
            self,
            original_terms: List[str],
            expanded_terms: List[str],
            synonym_weight: float
    ) -> List[float]:
        """
        加权语义搜索：原词权重1.0，同义词权重较低

        Args:
            original_terms: 原始搜索词
            expanded_terms: 扩展后的搜索词
            synonym_weight: 同义词权重

        Returns:
            每个文档的分数列表
        """
        doc_count = len(self._corpus)
        doc_scores = [0.0] * doc_count

        # 原词（权重 1.0）
        for term in original_terms:
            scores = self._bm25_model.get_scores([term])
            for i in range(doc_count):
                doc_scores[i] += scores[i]

        # 同义词（权重 synonym_weight）
        synonym_terms = [t for t in expanded_terms if t not in original_terms]
        for term in synonym_terms:
            scores = self._bm25_model.get_scores([term])
            for i in range(doc_count):
                doc_scores[i] += synonym_weight * scores[i]

        return doc_scores

    # 修改 _normalize_score 方法
    def _normalize_score(self, score: float, all_scores: List[float]) -> float:
        """归一化分数到 [0, 1] 区间 - 修复版"""
        if not all_scores:
            return 0.0

        # 过滤掉零分
        positive_scores = [s for s in all_scores if s > 0]
        if not positive_scores:
            return 0.0

        min_score = min(positive_scores)
        max_score = max(positive_scores)

        if max_score > min_score:
            # 线性归一化到 [0.1, 0.9] 区间，避免极端值
            norm_score = (score - min_score) / (max_score - min_score)
            # 缩放到 0.1-0.9 区间
            norm_score = 0.1 + norm_score * 0.8
        else:
            norm_score = 0.5

        return max(0.05, min(0.95, norm_score))

    def get_document_count(self) -> int:
        """获取语料库中的文档数量"""
        return len(self._corpus)


# 便捷函数
def create_bm25_retriever(
        variant: str = "okapi",
        use_jieba: bool = True,
        use_synonyms: bool = True
) -> BM25Retriever:
    """
    创建 BM25 检索器

    Args:
        variant: "okapi", "plus", "l"
        use_jieba: 是否使用 jieba 分词
        use_synonyms: 是否使用同义词扩展
    """
    variant_map = {
        "okapi": BM25Variant.OKAPI,
        "plus": BM25Variant.PLUS,
        "l": BM25Variant.L
    }
    return BM25Retriever(
        variant=variant_map.get(variant, BM25Variant.OKAPI),
        use_jieba=use_jieba,
        use_synonyms=use_synonyms
    )


__all__ = ['BM25Retriever', 'BM25Variant', 'create_bm25_retriever']