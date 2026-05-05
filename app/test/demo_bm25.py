# app/test/demo_bm25.py
"""BM25 检索测试脚本 - 支持同义词扩展"""

from rank_bm25 import BM25Okapi, BM25Plus, BM25L
import jieba
import re
from typing import List, Dict, Tuple


def jieba_tokenize(text: str) -> List[str]:
    """使用 jieba 分词"""
    # 移除标点符号
    text = re.sub(r'[^\u4e00-\u9fff\u0041-\u005a\u0061-\u007a0-9]', ' ', text)
    words = jieba.lcut(text)
    # 过滤单字符和空字符串
    words = [w for w in words if len(w) > 1]
    return words


class BM25WithSynonyms:
    """
    支持同义词的 BM25 检索器
    通过查询扩展实现 OR 语义
    """

    def __init__(self, corpus: List[str], tokenizer=jieba_tokenize):
        """
        初始化 BM25 检索器

        Args:
            corpus: 文档语料库
            tokenizer: 分词函数
        """
        self.tokenizer = tokenizer
        self.tokenized_corpus = [tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.original_corpus = corpus

        # 同义词词典
        self.synonyms = {
            "业务": ["业务范围", "主营业务", "经营范围", "核心业务"],
            "世运电路": ["世运电子", "世运", "公司"],
            "印制电路板": ["PCB", "电路板", "线路板", "印刷电路板"],
            "营收": ["收入", "营业收入", "销售额"],
            "利润": ["净利润", "盈利", "收益"],
            "汽车": ["车载", "车用", "汽车电子"],
            "特斯拉": ["Tesla", "特斯拉汽车"],
        }

    def expand_query(self, query: str) -> List[str]:
        """
        扩展查询：将同义词替换为 OR 形式

        Args:
            query: 原始查询，如 "世运电子的主要业务是什么？"

        Returns:
            扩展后的查询词列表，如 ['世运电子', '主要', '业务', '业务范围', '主营业务', '经营范围']
        """
        # 先分词
        tokens = self.tokenizer(query)

        # 扩展每个词
        expanded_tokens = []
        for token in tokens:
            # 添加原词
            expanded_tokens.append(token)
            # 添加同义词
            for word, syns in self.synonyms.items():
                if token == word or token in word or word in token:
                    expanded_tokens.extend(syns)
                # 检查同义词是否匹配
                for syn in syns:
                    if syn == token or token in syn:
                        expanded_tokens.extend([word] + syns)

        # 去重
        return list(set(expanded_tokens))

    def search_with_original_query(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """使用原始查询检索"""
        tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(tokens)

        # 排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append((idx, score, self.original_corpus[idx]))

        return results

    def search_with_expanded_query(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        使用扩展后的查询检索（模拟 OR 逻辑）

        方法：将扩展后的查询词分别计算分数，然后取最大值或加权和
        """
        expanded_tokens = self.expand_query(query)

        print(f"  扩展后词数: {len(expanded_tokens)}")
        print(f"  扩展词: {expanded_tokens[:10]}{'...' if len(expanded_tokens) > 10 else ''}")

        # 方法1：计算每个词的分数，取最大值（相当于 OR）
        all_scores = []
        for token in expanded_tokens:
            scores = self.bm25.get_scores([token])
            all_scores.append(scores)

        # 取最大值（OR 语义：只要匹配任意一个同义词即可）
        max_scores = [max(score[i] for score in all_scores) for i in range(len(self.original_corpus))]

        # 方法2：加权和（可选）
        # sum_scores = [sum(score[i] for score in all_scores) for i in range(len(self.original_corpus))]

        # 排序
        indexed_scores = list(enumerate(max_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append((idx, score, self.original_corpus[idx]))

        return results

    def search_with_weighted_synonyms(self, query: str, top_k: int = 5,
                                      synonym_weight: float = 0.8) -> List[Tuple[int, float, str]]:
        """
        使用加权同义词检索

        Args:
            query: 原始查询
            top_k: 返回数量
            synonym_weight: 同义词权重（0-1），原词权重为 1
        """
        # 分词
        original_tokens = self.tokenizer(query)

        # 扩展同义词
        expanded_map = {}  # {原词: [同义词列表]}
        for token in original_tokens:
            expanded_map[token] = [token]
            for word, syns in self.synonyms.items():
                if token == word or token in word or word in token:
                    expanded_map[token].extend(syns)
                for syn in syns:
                    if syn == token or token in syn:
                        expanded_map[token].extend([word] + syns)
            expanded_map[token] = list(set(expanded_map[token]))

        print(f"  原词及同义词: {expanded_map}")

        # 计算每个文档的加权分数
        doc_scores = [0.0] * len(self.original_corpus)

        for original_token, syn_tokens in expanded_map.items():
            # 原词分数（权重1.0）
            orig_scores = self.bm25.get_scores([original_token])

            # 同义词分数（权重 synonym_weight）
            syn_scores = [0.0] * len(self.original_corpus)
            for syn in syn_tokens:
                if syn != original_token:  # 跳过原词
                    scores = self.bm25.get_scores([syn])
                    for i in range(len(syn_scores)):
                        syn_scores[i] = max(syn_scores[i], scores[i])

            # 累加
            for i in range(len(doc_scores)):
                doc_scores[i] += orig_scores[i] + (synonym_weight * syn_scores[i])

        # 排序
        indexed_scores = list(enumerate(doc_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append((idx, score, self.original_corpus[idx]))

        return results


def test_bm25_with_synonyms():
    """测试带同义词的 BM25 检索"""
    print("=" * 70)
    print("BM25 同义词扩展测试（模拟 OR 语法）")
    print("=" * 70)

    # 测试文档
    corpus = [
        "世运电路的主要业务是印制电路板（PCB）的生产和销售",
        "公司专注于汽车电子领域的PCB产品，是特斯拉的核心供应商",
        "2023年上半年实现营业收入21.51亿元，同比增长46.85%",
        "公司产品结构持续优化，高附加值产品占比提升",
        "汽车PCB需求持续增长，新能源领域打开成长空间",
        "印制电路板是电子产品的基础组件，广泛应用于各个领域",
        "世运电子的主营业务范围包括多层板、HDI板的生产",
        "公司的经营范围涵盖印制线路板的研发制造",
    ]

    # 查询（使用 OR 语法）
    queries = [
        "世运电子的主要业务是什么？",
        "印制电路板",
        "公司的营收情况",
    ]

    bm25_syn = BM25WithSynonyms(corpus)

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"查询: {query}")
        print("=" * 70)

        # 1. 原始查询（无扩展）
        print("\n[方法1] 原始查询检索:")
        original_results = bm25_syn.search_with_original_query(query, top_k=3)
        for i, (idx, score, content) in enumerate(original_results, 1):
            print(f"  [{i}] 分数: {score:.4f}")
            print(f"      内容: {content[:80]}...")

        # 2. 扩展查询（最大值 OR）
        print("\n[方法2] 同义词扩展检索（最大值 OR）:")
        expanded_results = bm25_syn.search_with_expanded_query(query, top_k=3)
        for i, (idx, score, content) in enumerate(expanded_results, 1):
            print(f"  [{i}] 分数: {score:.4f}")
            print(f"      内容: {content[:80]}...")

        # 3. 加权同义词检索
        print("\n[方法3] 加权同义词检索（原词权重1.0，同义词权重0.8）:")
        weighted_results = bm25_syn.search_with_weighted_synonyms(query, top_k=3, synonym_weight=0.8)
        for i, (idx, score, content) in enumerate(weighted_results, 1):
            print(f"  [{i}] 分数: {score:.4f}")
            print(f"      内容: {content[:80]}...")


def test_query_rewrite_style():
    """测试类似 Query Rewriter 的改写风格"""
    print("\n" + "=" * 70)
    print("Query Rewriter 风格测试")
    print("=" * 70)

    corpus = [
        "世运电路主要业务为印制电路板的生产销售",
        "公司核心产品是PCB，用于汽车电子领域",
        "2023年中报显示营收21.51亿元",
        "特斯拉是公司重要客户，提供电动车PCB",
        "公司产品包括多层板、HDI板、软硬结合板",
    ]

    # 模拟 Query Rewriter 的输出
    rewritten_queries = [
        "世运电子的主要(业务 OR 业务范围 OR 主营业务 OR 经营范围)是什么？",
        "公司的(营收 OR 收入 OR 营业收入)情况",
        "印制电路板 OR PCB OR 电路板",
    ]

    bm25_syn = BM25WithSynonyms(corpus)

    for rewritten_query in rewritten_queries:
        print(f"\n{'=' * 60}")
        print(f"改写后查询: {rewritten_query}")
        print("=" * 60)

        # 提取 OR 中的词
        import re
        # 匹配 OR 两边的词
        or_pattern = r'\(([^)]+)\)'
        matches = re.findall(or_pattern, rewritten_query)

        # 构建扩展词列表
        expanded_terms = []
        query_without_or = rewritten_query
        for match in matches:
            terms = [t.strip() for t in match.split(' OR ')]
            expanded_terms.extend(terms)
            # 用第一个词替换 OR 表达式
            query_without_or = query_without_or.replace(f"({match})", terms[0])

        # 添加原始查询中的其他词
        base_terms = bm25_syn.tokenizer(query_without_or)
        all_terms = list(set(base_terms + expanded_terms))

        print(f"  提取的搜索词: {all_terms}")

        # 使用 OR 语义搜索
        doc_scores = [0.0] * len(corpus)
        for term in all_terms:
            scores = bm25_syn.bm25.get_scores([term])
            for i in range(len(doc_scores)):
                doc_scores[i] = max(doc_scores[i], scores[i])  # OR: 取最大值

        # 排序
        indexed_scores = list(enumerate(doc_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        print("\n检索结果:")
        for i, (idx, score) in enumerate(indexed_scores[:3], 1):
            if score > 0:
                print(f"  [{i}] 分数: {score:.4f}")
                print(f"      内容: {corpus[idx][:80]}...")


def test_hybrid_with_or_semantics():
    """测试混合检索中的 OR 语义实现"""
    print("\n" + "=" * 70)
    print("混合检索 OR 语义实现示例")
    print("=" * 70)

    corpus = [
        {"id": "1", "content": "世运电路主要业务是印制电路板PCB"},
        {"id": "2", "content": "公司专注于汽车电子领域，是特斯拉供应商"},
        {"id": "3", "content": "2023年中报营收21.51亿元，业绩增长"},
        {"id": "4", "content": "PCB产品用于新能源汽车，市场前景广阔"},
        {"id": "5", "content": "公司经营范围包括多层板HDI板生产"},
    ]

    # 用户原始问题
    original_question = "世运电子的主要业务是什么？"

    # Query Rewriter 改写后的查询（带 OR 语法）
    rewritten_query = "世运电子的主要(业务 OR 业务范围 OR 主营业务 OR 经营范围)是什么？"

    print(f"原始问题: {original_question}")
    print(f"改写后: {rewritten_query}")
    print()

    # 解析 OR 表达式
    def parse_or_query(query: str) -> List[str]:
        """解析 OR 查询，返回搜索词列表"""
        # 提取 OR 组
        import re
        or_pattern = r'\(([^)]+)\)'

        all_terms = []

        # 处理 OR 组
        for match in re.findall(or_pattern, query):
            terms = [t.strip() for t in match.split(' OR ')]
            all_terms.extend(terms)
            # 从查询中移除已处理的 OR 组
            query = query.replace(f"({match})", "")

        # 添加剩余的词
        import jieba
        remaining_terms = jieba.lcut(query)
        all_terms.extend([t for t in remaining_terms if len(t) > 1])

        return list(set(all_terms))

    # 获取搜索词
    search_terms = parse_or_query(rewritten_query)
    print(f"搜索词 (OR 语义): {search_terms}")
    print()

    # BM25 检索（OR 语义：取最大值）
    tokenized_corpus = [jieba_tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # 计算每个文档的 BM25 分数（OR 语义）
    doc_scores = []
    for i, doc in enumerate(corpus):
        max_score = 0.0
        for term in search_terms:
            score = bm25.get_scores([term])[i]
            max_score = max(max_score, score)
        doc_scores.append((doc["id"], max_score, doc["content"]))

    # 排序
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    print("BM25 检索结果 (OR 语义):")
    for rank, (doc_id, score, content) in enumerate(doc_scores[:3], 1):
        print(f"  [{rank}] 文档{doc_id}, 分数: {score:.4f}")
        print(f"      内容: {content[:60]}...")

    print("\n" + "-" * 50)
    print("说明：")
    print("  1. BM25 本身不支持 OR 语法")
    print("  2. 通过解析 OR 表达式，分别计算每个词的分数")
    print("  3. 取最大值实现 OR 语义（匹配任一关键词即可）")
    print("  4. 也可以使用加权和（给同义词较低权重）")


if __name__ == "__main__":
    # 测试同义词扩展
    test_bm25_with_synonyms()

    # 测试 Query Rewriter 风格
    test_query_rewrite_style()

    # 测试混合检索中的 OR 语义
    test_hybrid_with_or_semantics()