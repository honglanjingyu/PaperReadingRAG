# RAG3.py - 入口文件，只调用不实现
"""
本文件现在只是一个调用入口，所有功能通过导入 app/service/core 模块实现

处理顺序：
1. 数据加载
2. 布局识别
3. 连接跨页内容
4. 数据清洗
5. 智能分块
6. 向量化
7. 向量存储
8. 用户问题向量化
9. 相似度搜索
10.增强搜索
"""

import os
import sys
from typing import Optional, List
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从 deepdoc 模块导入 RAG1 的完整流程
from app.service.core.deepdoc import (
    # 数据结构
    TextBlock,
    TableBlock,
    PageContent,
    ParsedDocument,
    LayoutType,

    # 模块类 - RAG1 的核心处理流程
    DataLoader,           # 1. 数据加载
    LayoutRecognizer,     # 2. 布局识别
    CrossPageConnector,   # 3. 连接跨页内容
    DocumentParser,       # 整合上述三步 + 数据清洗

    # 清洗器
    DataCleaner,
    CleaningPipeline,

    # 便捷函数
    parse_document,       # 完整解析（数据加载->布局识别->跨页连接->清洗）
    parse_document_to_text,
    clean_text,
)

# 从 chunking 模块导入分块功能
from app.service.core.chunking import (
    ChunkManager,
    ChunkProcessor,
    RecursiveChunkerSimple,
    create_chunker,
    chunk_text_to_chunks,
    chunk_text_simple,
    get_chunk_statistics,
    Chunk,
    ChunkStrategy,
)

# 从 embedding 模块导入向量化功能
from app.service.core.embedding import (
    VectorChunk,
    VectorizationService,
    vectorize_chunks,
    vectorize_text,
    EmbeddingManager,
    EmbeddingType,
    get_embedding_manager,
    get_embedding_service,
)

# 从 vector_store 模块导入向量存储功能
from app.service.core.vector_store import (
    ESVectorStore,
    VectorStorageService,
    get_vector_storage_service,
    get_vector_search_service,
)


# ==================== 全局测试问题列表 ====================

TEST_QUESTIONS = [
    "世运电子的主要业务是什么？",
    # "公司2023年中报的营收情况如何？",
    # "请分析世运电子的盈利能力",
    # "公司的主要客户有哪些？",
    # "世运电子的竞争优势是什么？",
    # "世运电子的盈利能力怎么样？",
]

# ==================== 便捷函数：完整的文档处理流程 ====================

def process_document(
    file_path: str,
    chunk_size: int = 256,
    enable_vectorization: bool = True,
    enable_storage: bool = False,
    model_type: str = None,
    from_page: int = 0,
    to_page: int = None,
    index_name: str = None,
    verbose: bool = False
) -> List[VectorChunk]:
    """
    完整的文档处理流程：
    RAG1流程：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗
    RAG2流程：智能分块 -> 向量化 -> 向量存储
    RAG3流程: 用户问题向量化 -> 相似度搜索 -> 增强搜索

    Args:
        file_path: 文件路径
        chunk_size: 分块大小（token数）
        enable_vectorization: 是否启用向量化
        enable_storage: 是否存储到向量数据库
        model_type: 模型类型 ('remote' 或 'local')
        from_page: 起始页
        to_page: 结束页
        index_name: ES索引名称
        verbose: 是否打印详细信息

    Returns:
        VectorChunk 列表
    """
    if verbose:
        print("=" * 70)
        print(f"处理文档: {os.path.basename(file_path)}")
        print("=" * 70)
        print("\n" + "-" * 70)
        print("\nRAG1 流程:")
        print("  1. 数据加载")
        print("  2. 布局识别")
        print("  3. 连接跨页内容")
        print("  4. 数据清洗")
        print("-" * 70)

    # ========== RAG1 完整流程：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗 ==========
    parser = DocumentParser()
    parsed = parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=True,
        verbose=verbose
    )

    if not parsed.cleaned_text:
        if verbose:
            print("错误: 未能提取文本内容")
        return []

    if verbose:
        print(f"\nRAG1 处理结果:")
        print(f"  文件名: {parsed.file_name}")
        print(f"  文件类型: {parsed.file_type}")
        print(f"  总页数: {parsed.total_pages}")
        print(f"  清洗后文本长度: {len(parsed.cleaned_text)} 字符")

    if verbose:
        print("\n" + "-" * 70)
        print("RAG2 流程:")
        print("  5. 智能分块")
        print("  6. 向量化")
        print("  7. 向量存储")
        print("-" * 70)

    # ========== RAG2 功能：智能分块 ==========
    if verbose:
        print("\n[5/7] 智能分块...")

    chunks = chunk_text_to_chunks(
        parsed.cleaned_text,
        chunk_size=chunk_size,
        metadata={
            'source': parsed.file_name,
            'file_type': parsed.file_type,
            'total_pages': parsed.total_pages
        },
        strategy='recursive'
    )

    if verbose:
        stats = get_chunk_statistics(chunks)
        print(f"  生成 {stats['total_chunks']} 个块")
        print(f"  Token统计: 最小={stats['min_token_count']}, "
              f"最大={stats['max_token_count']}, "
              f"平均={stats['avg_token_count']:.1f}")

    # ========== RAG2 功能：向量化 ==========
    vector_chunks = []
    if enable_vectorization and chunks:
        if verbose:
            print("\n[6/7] 向量化处理...")

        try:
            # 转换为 VectorChunk
            import hashlib
            vector_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{i}_{chunk.content[:100]}".encode()).hexdigest()[:16]
                vector_chunks.append(VectorChunk(
                    id=f"chunk_{i}_{chunk_id}",
                    content=chunk.content,
                    metadata={
                        **chunk.metadata,
                        'chunk_index': i,
                        'token_count': chunk.token_count
                    },
                    token_count=chunk.token_count,
                    chunk_index=i
                ))

            # 向量化
            vec_service = VectorizationService(model_type)
            vector_chunks = vec_service.vectorize_chunks(vector_chunks)

            if verbose:
                vectorized_count = len([c for c in vector_chunks if c.vector])
                print(f"  向量化完成: {vectorized_count}/{len(vector_chunks)} 个块")
                print(f"  向量维度: {vec_service.dimension}")

        except Exception as e:
            if verbose:
                print(f"  向量化失败: {e}")
            vector_chunks = []

    # ========== RAG2 功能：向量存储 ==========
    if enable_storage and vector_chunks:
        if verbose:
            print("\n[7/7] 存储到向量数据库...")

        try:
            storage_service = get_vector_storage_service()
            index = index_name or os.getenv("ES_INDEX_NAME", "rag_documents")
            inserted = storage_service.store_vector_chunks(vector_chunks, index, parsed.file_name)

            if verbose:
                print(f"  存储完成: {inserted}/{len(vector_chunks)} 条")

        except Exception as e:
            if verbose:
                print(f"  存储失败: {e}")

    return vector_chunks if vector_chunks else chunks


def parse_with_rag1_only(
    file_path: str,
    from_page: int = 0,
    to_page: int = None,
    enable_cleaning: bool = True,
    verbose: bool = False
) -> ParsedDocument:
    """
    仅执行 RAG1 流程：数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗

    Args:
        file_path: 文件路径
        from_page: 起始页
        to_page: 结束页
        enable_cleaning: 是否启用清洗
        verbose: 是否打印详细信息

    Returns:
        ParsedDocument 对象
    """
    parser = DocumentParser()
    return parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=enable_cleaning,
        verbose=verbose
    )


def chunk_document(
    file_path: str,
    chunk_size: int = 256,
    from_page: int = 0,
    to_page: int = None,
    verbose: bool = False
) -> List[Chunk]:
    """
    仅执行：RAG1完整流程 + 智能分块（不含向量化和存储）

    Args:
        file_path: 文件路径
        chunk_size: 分块大小
        from_page: 起始页
        to_page: 结束页
        verbose: 是否打印详细信息

    Returns:
        Chunk 列表
    """
    # RAG1 完整流程
    parser = DocumentParser()
    parsed = parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=True,
        verbose=verbose
    )

    if not parsed.cleaned_text:
        return []

    # 分块
    return chunk_text_to_chunks(
        parsed.cleaned_text,
        chunk_size=chunk_size,
        metadata={'source': parsed.file_name, 'file_type': parsed.file_type}
    )


def vectorize_chunk_texts(
    texts: List[str],
    metadata_list: List[dict] = None,
    model_type: str = None
) -> List[VectorChunk]:
    """向量化文本列表"""
    chunks = []
    for i, text in enumerate(texts):
        import hashlib
        chunk_id = hashlib.md5(f"{i}_{text[:100]}".encode()).hexdigest()[:16]
        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
        chunks.append(VectorChunk(
            id=f"chunk_{i}_{chunk_id}",
            content=text,
            metadata=metadata
        ))

    return vectorize_chunks(chunks, model_type)


def get_rag1_processing_stats(file_path: str, from_page: int = 0, to_page: int = None) -> dict:
    """
    获取 RAG1 处理统计信息

    Args:
        file_path: 文件路径
        from_page: 起始页
        to_page: 结束页

    Returns:
        统计信息字典
    """
    parser = DocumentParser()
    parsed = parser.parse(
        file_path,
        from_page=from_page,
        to_page=to_page or 100000,
        enable_cleaning=True,
        verbose=False
    )

    # 获取统计信息
    total_text_blocks = sum(len(p.text_blocks) for p in parsed.pages)
    total_tables = sum(len(p.tables) for p in parsed.pages)

    return {
        'file_name': parsed.file_name,
        'file_type': parsed.file_type,
        'total_pages': parsed.total_pages,
        'raw_text_length': len(parsed.cleaned_text) if parsed.cleaned_text else 0,
        'cleaned_text_length': len(parsed.cleaned_text),
        'total_text_blocks': total_text_blocks,
        'total_tables': total_tables,
    }


# 向后兼容的类
class RecursiveChunker:
    """向后兼容的分块器类"""

    def __init__(self, chunk_token_num: int = 256):
        self.chunker = RecursiveChunkerSimple(chunk_token_num=chunk_token_num)

    def chunk(self, text: str) -> List[str]:
        return self.chunker.chunk(text)


# 导出所有类和函数
__all__ = [
    # 从 deepdoc 导入（RAG1 完整功能）
    'TextBlock', 'TableBlock', 'PageContent', 'ParsedDocument', 'LayoutType',
    'DataLoader', 'LayoutRecognizer', 'CrossPageConnector', 'DocumentParser',
    'DataCleaner', 'CleaningPipeline',
    'parse_document', 'parse_document_to_text', 'clean_text',

    # 从 chunking 导入
    'ChunkManager', 'ChunkProcessor', 'RecursiveChunkerSimple',
    'create_chunker', 'chunk_text_to_chunks', 'chunk_text_simple',
    'get_chunk_statistics', 'Chunk', 'ChunkStrategy',

    # 从 embedding 导入
    'VectorChunk', 'VectorizationService', 'vectorize_chunks', 'vectorize_text',
    'EmbeddingManager', 'EmbeddingType', 'get_embedding_manager', 'get_embedding_service',

    # 从 vector_store 导入
    'ESVectorStore', 'VectorStorageService', 'get_vector_storage_service',

    # 便捷函数
    'process_document',           # 完整流程：RAG1 + 分块 + 向量化 + 存储
    'parse_with_rag1_only',       # 仅 RAG1 流程
    'chunk_document',             # RAG1 + 分块
    'vectorize_chunk_texts',      # 仅向量化
    'get_rag1_processing_stats',  # RAG1 统计信息
    'RecursiveChunker',

    # 全局测试问题
    'TEST_QUESTIONS',
]


# ==================== 新增功能：用户问题向量化 ====================

def vectorize_user_question(
        question: str,
        model_type: str = None,
        verbose: bool = True
) -> dict:
    """
    8. 用户问题 - 接收用户输入的问题
    9. 问题向量 - 将用户输入的问题向量化

    Args:
        question: 用户问题文本
        model_type: 模型类型 ('remote' 或 'local')，默认从环境变量读取
        verbose: 是否打印详细信息

    Returns:
        dict: 包含问题文本、向量、向量维度、模型信息的结果字典
    """
    if verbose:
        print("\n[8/10] 接收用户问题...")
        print(f"\n用户问题: {question}")
        print(f"问题长度: {len(question)} 字符")

    # 9. 将用户输入的问题向量化
    if verbose:
        print("\n问题向量化...")

    try:
        # 使用 embedding 服务生成向量
        from app.service.core.embedding import VectorizationService

        # 确定模型类型
        if model_type is None:
            model_type = os.getenv("EMBEDDING_TYPE", "remote")

        vec_service = VectorizationService(model_type)

        # 生成问题向量
        question_vector = vec_service.manager.generate_embedding(question)

        if question_vector is None:
            print(f"  问题向量化失败")
            return None

        if verbose:
            print(f"  问题向量化成功")
            print(f"  向量维度: {len(question_vector)}")
            print(f"  向量前5维预览: {question_vector[:5]}...")

            # 获取模型信息
            model_info = vec_service.get_model_info()
            print(f"  模型类型: {model_info.get('type', 'unknown')}")
            print(f"  模型名称: {model_info.get('model_name', 'unknown')}")

        result = {
            "success": True,
            "question": question,
            "question_length": len(question),
            "vector": question_vector,
            "vector_dimension": len(question_vector),
            "model_type": model_type,
            "model_info": vec_service.get_model_info()
        }

        return result

    except Exception as e:
        if verbose:
            print(f"  问题向量化失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e)
        }


def test_user_question_vectorization(questions: List[str] = None):
    """
    测试用户问题向量化功能

    Args:
        questions: 要测试的问题列表，默认使用全局 TEST_QUESTIONS
    """
    if questions is None:
        questions = TEST_QUESTIONS

    print("\n默认测试问题:")
    for q in questions[:3]:  # 只显示前3个
        print(f"  - {q}")
    if len(questions) > 3:
        print(f"  ... 共 {len(questions)} 个问题")

    results = []
    for question in questions:
        result = vectorize_user_question(question=question, verbose=False)

        if result and result.get("success"):
            print(f"\n✓ 问题: {question}")
            print(f"  向量维度: {result['vector_dimension']}")
            print(f"  向量前5维: {result['vector'][:5]}...")
            results.append(result)
        else:
            print(f"\n✗ 问题: {question}")
            print(f"  失败: {result.get('error', '未知错误') if result else '未知错误'}")

    return results


# ==================== 新增功能：相似度搜索 ====================

def search_similar_documents(
    question: str,
    es_index_name: str = None,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    model_type: str = None,
    verbose: bool = True
) -> dict:
    """
    10. 相似度搜索：在向量数据库中召回最相关的 Top-K 个文档块

    处理流程：
    8. 用户问题 - 接收用户输入的问题
    9. 问题向量 - 将用户输入的问题向量化
    10. 相似度搜索 - 在向量数据库中召回最相关的 Top-K 个文档块

    Args:
        question: 用户问题
        index_name: ES索引名称，默认从环境变量读取
        top_k: 返回的文档块数量 (Top-K)
        similarity_threshold: 相似度阈值
        model_type: 模型类型 ('remote' 或 'local')
        verbose: 是否打印详细信息

    Returns:
        dict: 包含问题、向量、搜索结果的结果字典
    """
    # 8. 接收用户问题
    if verbose:
        print(f"\n[8/10] 用户问题: {question}")
        print(f"问题长度: {len(question)} 字符")

    # 9. 问题向量化
    if verbose:
        print("\n问题向量化...")

    try:
        from app.service.core.embedding import VectorizationService

        if model_type is None:
            model_type = os.getenv("EMBEDDING_TYPE", "remote")

        vec_service = VectorizationService(model_type)

        # 生成问题向量
        question_vector = vec_service.manager.generate_embedding(question)

        if question_vector is None:
            return {
                "success": False,
                "question": question,
                "error": "问题向量化失败"
            }

        if verbose:
            print(f"  向量维度: {len(question_vector)}")
            print(f"  向量前5维预览: {question_vector[:5]}...")

    except Exception as e:
        if verbose:
            print(f"  向量化失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e)
        }

    # 10. 相似度搜索：在向量数据库中召回最相关的 Top-K 个文档块
    if verbose:
        print(f"\n[9/10] 相似度搜索 (Top-K={top_k})...")

    try:
        from app.service.core.vector_store import get_vector_search_service

        # 确定索引名称
        if es_index_name is None:
            es_index_name = os.getenv("ES_INDEX_NAME", "rag_documents")

        search_service = get_vector_search_service()
        results = search_service.similarity_search(
            query_vector=question_vector,
            index_name=es_index_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        if verbose:
            print(f"  索引名称: {es_index_name}")
            print(f"  召回数量: {len(results)}/{top_k}")
            print(f"  相似度阈值: {similarity_threshold}")

        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "score": result.get("_score", 0),
                "content": result.get("content_with_weight", result.get("content", "")),
                "document_name": result.get("docnm", result.get("docnm_kwd", "")),
                "chunk_id": result.get("_id", result.get("id", "")),
                "metadata": {k: v for k, v in result.items()
                           if k not in ["content", "content_with_weight", "_score", "_id", "id"]}
            })

        if verbose and formatted_results:
            print("\n" + "-" * 70)
            print("搜索结果详情:")
            print("-" * 70)
            for res in formatted_results:
                print(f"\n  [排名 {res['rank']}] 相似度: {res['score']:.4f}")
                print(f"  文档: {res['document_name']}")
                content_preview = res['content'][:200].replace('\n', ' ')
                print(f"  内容预览: {content_preview}...")

        return {
            "success": True,
            "question": question,
            "question_length": len(question),
            "query_vector": question_vector,
            "vector_dimension": len(question_vector),
            "model_type": model_type,
            "model_info": vec_service.get_model_info(),
            "index_name": es_index_name,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "total_recalled": len(results),
            "results": formatted_results
        }

    except Exception as e:
        if verbose:
            print(f"  相似度搜索失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e)
        }


def test_similarity_search(questions: List[str] = None):
    """
    测试相似度搜索功能

    Args:
        questions: 要测试的问题列表，默认使用全局 TEST_QUESTIONS
    """
    if questions is None:
        questions = TEST_QUESTIONS

    # 从环境变量读取配置
    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    top_k = int(os.getenv("SIMILARITY_TOP_K", "5"))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    print(f"\n配置信息:")
    print(f"  索引名称: {index_name}")
    print(f"  Top-K: {top_k}")
    print(f"  相似度阈值: {similarity_threshold}")

    all_results = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"测试 {i}: {question}")
        print("=" * 70)

        result = search_similar_documents(
            question=question,
            es_index_name=index_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            verbose=True
        )

        if result.get("success"):
            print(f"\n✓ 测试 {i} 成功")
            print(f"  召回文档块数: {result['total_recalled']}")
            all_results.append(result)
        else:
            print(f"\n✗ 测试 {i} 失败: {result.get('error')}")

    return all_results


# ==================== 新增功能：增强检索 ====================

def enhanced_search_with_hybrid_and_rerank(
    question: str,
    index_name: str = None,
    top_k: int = 5,
    keyword_weight: float = 0.3,
    vector_weight: float = 0.7,
    enable_rerank: bool = True,
    enable_query_rewrite: bool = False,
    similarity_threshold: float = 0.3,
    verbose: bool = True
) -> dict:
    """
    增强检索：混合检索 + 重排序 + Query改写

    功能说明：
    1. 混合检索 - 同时使用关键词检索和向量检索
    2. 重排序 - 使用Cross-Encoder或向量相似度对结果重新排序
    3. Query改写 - 同义词扩展，提升召回率

    Args:
        question: 用户问题
        index_name: ES索引名称
        top_k: 返回数量
        keyword_weight: 关键词检索权重 (0-1)
        vector_weight: 向量检索权重 (0-1)
        enable_rerank: 是否启用重排序
        enable_query_rewrite: 是否启用Query改写
        similarity_threshold: 相似度阈值
        verbose: 是否打印详细信息

    Returns:
        增强检索结果
    """
    if verbose:
        print("\n" + "=" * 70)
        print("增强检索模式 (混合检索 + 重排序 + Query改写)")
        print("=" * 70)
        print(f"\n原始问题: {question}")

    # 确定索引名称
    if index_name is None:
        index_name = os.getenv("ES_INDEX_NAME", "rag_documents")

    # 1. Query改写（可选）
    rewritten_query = question
    if enable_query_rewrite:
        if verbose:
            print("\nQuery改写...")

        try:
            from app.service.core.retrieval import QueryRewriter
            rewriter = QueryRewriter()
            rewritten_query = rewriter.rewrite(question, strategy='synonym')

            # 生成子查询用于多路召回
            sub_queries = rewriter.generate_sub_queries(question, max_queries=3)

            if verbose:
                print(f"  改写后问题: {rewritten_query}")
                print(f"  子查询: {sub_queries}")
        except Exception as e:
            if verbose:
                print(f"  Query改写失败: {e}，使用原始查询")
            rewritten_query = question
            sub_queries = [question]
    else:
        sub_queries = [question]

    # 2. 检查索引
    if verbose:
        print("\n检查索引...")

    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(index_name)

        if not index_exists:
            return {
                "success": False,
                "error": f"索引 '{index_name}' 不存在，请先处理文档"
            }

        doc_count = search_service.es_store.get_document_count(index_name)
        if verbose:
            print(f"  索引名称: {index_name}")
            print(f"  文档块数量: {doc_count}")

        if doc_count == 0:
            return {
                "success": False,
                "error": f"索引 '{index_name}' 为空"
            }

    except Exception as e:
        return {"success": False, "error": str(e)}

    # 3. 混合检索
    if verbose:
        print("\n混合检索...")
        print(f"  关键词权重: {keyword_weight}")
        print(f"  向量权重: {vector_weight}")

    try:
        from app.service.core.retrieval import HybridRetriever
        hybrid_retriever = HybridRetriever()

        # 使用改写后的查询进行混合检索
        hybrid_results = hybrid_retriever.hybrid_search(
            query=rewritten_query,
            index_name=index_name,
            top_k=top_k * 2,  # 多召回一些用于重排序
            keyword_weight=keyword_weight,
            vector_weight=vector_weight,
            similarity_threshold=similarity_threshold
        )

        if verbose:
            print(f"  混合检索召回: {len(hybrid_results)} 个块")

    except Exception as e:
        if verbose:
            print(f"  混合检索失败: {e}，回退到向量检索")

        # 回退到向量检索
        from app.service.core.embedding import get_embedding_manager
        from app.service.core.vector_store import get_vector_search_service

        embedding_manager = get_embedding_manager()
        query_vector = embedding_manager.generate_embedding(rewritten_query)

        if query_vector:
            search_service = get_vector_search_service()
            hybrid_results = search_service.similarity_search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k * 2,
                similarity_threshold=similarity_threshold
            )
        else:
            hybrid_results = []

    # 4. 重排序（可选）
    if enable_rerank and hybrid_results:
        if verbose:
            print("\n[10/10] 重排序...")

        try:
            from app.service.core.retrieval import Reranker
            reranker = Reranker()

            reranked_results = reranker.rerank(
                query=rewritten_query,
                documents=hybrid_results,
                top_k=top_k
            )

            if verbose:
                print(f"  重排序完成: {len(reranked_results)} 个块")
            final_results = reranked_results

        except Exception as e:
            if verbose:
                print(f"  重排序失败: {e}，使用原始排序")
            final_results = hybrid_results[:top_k]
    else:
        final_results = hybrid_results[:top_k]

    # 格式化结果
    formatted_results = []
    for i, result in enumerate(final_results, 1):
        formatted_results.append({
            "rank": i,
            "score": result.get("final_score", result.get("_score", result.get("rerank_score", 0))),
            "vector_score": result.get("vector_score", 0),
            "keyword_score": result.get("keyword_score", 0),
            "rerank_score": result.get("rerank_score", 0),
            "content": result.get("content_with_weight", result.get("content", "")),
            "document_name": result.get("docnm", result.get("docnm_kwd", "")),
            "chunk_id": result.get("_id", result.get("id", "")),
            "search_type": result.get("_search_types", ["unknown"])
        })

    if verbose and formatted_results:
        print("\n" + "-" * 70)
        print("增强检索结果详情:")
        print("-" * 70)
        for res in formatted_results:
            print(f"\n  [排名 {res['rank']}]")
            print(f"  综合分数: {res['score']:.4f}")
            print(f"  向量分: {res['vector_score']:.4f} | 关键词分: {res['keyword_score']:.4f}")
            print(f"  文档: {res['document_name']}")
            content_preview = res['content'][:200].replace('\n', ' ')
            print(f"  内容预览: {content_preview}...")

    return {
        "success": True,
        "question": question,
        "rewritten_query": rewritten_query if enable_query_rewrite else None,
        "index_name": index_name,
        "top_k": top_k,
        "keyword_weight": keyword_weight,
        "vector_weight": vector_weight,
        "enable_rerank": enable_rerank,
        "enable_query_rewrite": enable_query_rewrite,
        "total_recalled": len(hybrid_results),
        "total_returned": len(formatted_results),
        "results": formatted_results
    }


def test_enhanced_retrieval(questions: List[str] = None):
    """测试增强检索功能

    Args:
        questions: 要测试的问题列表，默认使用全局 TEST_QUESTIONS
    """
    if questions is None:
        questions = TEST_QUESTIONS

    print("\n" + "=" * 70)
    print("测试增强检索功能")
    print("=" * 70)

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    top_k = int(os.getenv("SIMILARITY_TOP_K", "5"))

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"测试 {i}: {question}")
        print("=" * 70)

        # 对比不同模式
        modes = [
            ("向量检索", False, False, False),
            ("混合检索", True, False, False),
            ("混合检索+重排序", True, True, False),
            ("完整增强检索", True, True, True),
        ]

        for mode_name, use_hybrid, use_rerank, use_rewrite in modes:
            print(f"\n--- {mode_name} ---")

            result = enhanced_search_with_hybrid_and_rerank(
                question=question,
                index_name=index_name,
                top_k=top_k,
                keyword_weight=0.3 if use_hybrid else 1.0,
                vector_weight=0.7 if use_hybrid else 1.0,
                enable_rerank=use_rerank,
                enable_query_rewrite=use_rewrite,
                verbose=False
            )

            if result.get("success"):
                print(f"  召回数量: {result['total_returned']}")
                if result['results']:
                    best = result['results'][0]
                    print(f"  最佳匹配: {best['document_name']}")
                    print(f"  综合分数: {best['score']:.4f}")
            else:
                print(f"  失败: {result.get('error')}")


def compare_search_methods(questions: List[str] = None):
    """对比传统检索和增强检索的效果

    Args:
        questions: 要测试的问题列表，默认使用全局 TEST_QUESTIONS
    """
    if questions is None:
        questions = TEST_QUESTIONS

    print("\n" + "=" * 70)
    print("传统检索 vs 增强检索 对比")
    print("=" * 70)

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    top_k = int(os.getenv("SIMILARITY_TOP_K", "5"))

    comparison_results = []

    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 50)

        # 传统向量检索
        from app.service.core.vector_store import get_vector_search_service
        from app.service.core.embedding import get_embedding_manager

        embedding_manager = get_embedding_manager()
        query_vector = embedding_manager.generate_embedding(question)

        if query_vector:
            search_service = get_vector_search_service()
            traditional_results = search_service.similarity_search(
                query_vector=query_vector,
                index_name=index_name,
                top_k=top_k,
                similarity_threshold=0.3
            )
            traditional_avg_score = sum(r.get('_score', 0) for r in traditional_results) / len(traditional_results) if traditional_results else 0
            traditional_count = len(traditional_results)
        else:
            traditional_avg_score = 0
            traditional_count = 0

        # 增强检索
        enhanced_result = enhanced_search_with_hybrid_and_rerank(
            question=question,
            index_name=index_name,
            top_k=top_k,
            keyword_weight=0.3,
            vector_weight=0.7,
            enable_rerank=True,
            enable_query_rewrite=True,
            verbose=False
        )

        enhanced_count = enhanced_result.get('total_returned', 0) if enhanced_result.get('success') else 0
        enhanced_avg_score = 0
        if enhanced_result.get('success') and enhanced_result['results']:
            enhanced_avg_score = sum(r['score'] for r in enhanced_result['results']) / len(enhanced_result['results'])

        print(f"  传统向量检索: 召回 {traditional_count} 块, 平均分数 {traditional_avg_score:.4f}")
        print(f"  增强检索: 召回 {enhanced_count} 块, 平均分数 {enhanced_avg_score:.4f}")

        comparison_results.append({
            "question": question,
            "traditional": {"count": traditional_count, "avg_score": traditional_avg_score},
            "enhanced": {"count": enhanced_count, "avg_score": enhanced_avg_score}
        })

    # 汇总对比
    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)

    avg_traditional_count = sum(r["traditional"]["count"] for r in comparison_results) / len(comparison_results)
    avg_enhanced_count = sum(r["enhanced"]["count"] for r in comparison_results) / len(comparison_results)
    avg_traditional_score = sum(r["traditional"]["avg_score"] for r in comparison_results) / len(comparison_results)
    avg_enhanced_score = sum(r["enhanced"]["avg_score"] for r in comparison_results) / len(comparison_results)

    print(f"\n平均召回数量: 传统={avg_traditional_count:.1f} | 增强={avg_enhanced_count:.1f} | 提升={avg_enhanced_count - avg_traditional_count:.1f}")
    print(f"平均相似度: 传统={avg_traditional_score:.4f} | 增强={avg_enhanced_score:.4f} | 提升={avg_enhanced_score - avg_traditional_score:.4f}")

    return comparison_results


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("运行所有测试")
    print("=" * 70)

    # 1. 用户问题向量化测试
    print("\n" + "=" * 70)
    print("1. 用户问题向量化测试")
    print("=" * 70)
    test_user_question_vectorization()

    # 2. 相似度搜索测试
    print("\n" + "=" * 70)
    print("2. 相似度搜索测试")
    print("=" * 70)

    index_name = os.getenv("ES_INDEX_NAME", "rag_documents")
    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(index_name)

        if index_exists and search_service.es_store.get_document_count(index_name) > 0:
            test_similarity_search()
        else:
            print(f"\n⚠ 索引 '{index_name}' 不存在或为空，跳过相似度搜索测试")
    except Exception as e:
        print(f"\n检查索引时出错: {e}")

    # 3. 增强检索测试
    print("\n" + "=" * 70)
    print("3. 增强检索测试")
    print("=" * 70)

    try:
        from app.service.core.vector_store import get_vector_search_service
        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(index_name)

        if index_exists and search_service.es_store.get_document_count(index_name) > 0:
            # 使用前3个问题进行测试
            test_enhanced_retrieval(TEST_QUESTIONS[:3])
        else:
            print(f"\n⚠ 索引 '{index_name}' 不存在或为空，跳过增强检索测试")
    except Exception as e:
        print(f"\n增强检索测试失败: {e}")


# ==================== 演示 ====================
if __name__ == "__main__":
    print("\n处理流程:")
    print("  RAG1 流程: 数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗")
    print("  RAG2 流程: 智能分块 -> 向量化 -> 向量存储")
    print("  RAG3 流程: 用户问题向量化 -> 相似度搜索 -> 增强搜索")
    print("=" * 70)

    # 从环境变量读取配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
    ENABLE_VECTORIZATION = os.getenv("ENABLE_VECTORIZATION", "true").lower() == "true"
    ENABLE_STORAGE = os.getenv("ENABLE_STORAGE", "false").lower() == "true"
    MODEL_TYPE = os.getenv("EMBEDDING_TYPE", "local")

    # 定义索引名称（供后续使用）
    INDEX_NAME = os.getenv("ES_INDEX_NAME", "rag_documents")
    TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    print(f"\n配置信息:")
    print(f"  分块大小: {CHUNK_SIZE} tokens")
    print(f"  向量化: {'启用' if ENABLE_VECTORIZATION else '禁用'}")
    print(f"  存储: {'启用' if ENABLE_STORAGE else '禁用'}")
    if ENABLE_VECTORIZATION:
        print(f"  模型类型: {MODEL_TYPE}")
    print(f"  索引名称: {INDEX_NAME}")
    print(f"  Top-K: {TOP_K}")
    print(f"  相似度阈值: {SIMILARITY_THRESHOLD}")

    print(f"\n全局测试问题列表 ({len(TEST_QUESTIONS)} 个):")
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"  {i}. {q}")

    # 测试 PDF 文件
    pdf_file = "【兴证电子】世运电路2023中报点评.pdf"

    if os.path.exists(pdf_file):
        print(f"\n测试文档: {pdf_file}")
        print("-" * 70)

        # 使用完整流程处理
        result = process_document(
            file_path=pdf_file,
            chunk_size=CHUNK_SIZE,
            enable_vectorization=ENABLE_VECTORIZATION,
            enable_storage=ENABLE_STORAGE,
            model_type=MODEL_TYPE,
            from_page=0,
            to_page=5,
            verbose=True
        )

        # 显示详细结果
        print(f"\n处理结果摘要:")
        print(f"  生成块数: {len(result)}")

        if result:
            vectorized = [c for c in result if hasattr(c, 'vector') and c.vector]
            if vectorized:
                print(f"  向量化块数: {len(vectorized)}")
                print(f"  向量维度: {len(vectorized[0].vector)}")

            # 显示前3个块预览
            print(f"\n前3个块预览:")
            for i, chunk in enumerate(result[:3]):
                preview = chunk.content[:150].replace('\n', ' ')
                token_count = chunk.token_count if hasattr(chunk, 'token_count') else 0
                print(f"\n  块 {i + 1} (Token: {token_count}):")
                print(f"    {preview}...")

            # 如果启用了向量化，显示向量预览
            if vectorized and len(vectorized) > 0:
                vec_preview = vectorized[0].vector[:5] if vectorized[0].vector else []
                print(f"\n向量预览（前5维）:")
                print(f"    {vec_preview}...")

    else:
        print(f"\n测试文件不存在: {pdf_file}")
        print("请将 PDF 文件放在当前目录下")

        # 尝试查找 PDF 文件
        print("\n当前目录下的 PDF 文件:")
        for f in os.listdir('.'):
            if f.lower().endswith('.pdf'):
                print(f"  - {f}")

    # ==================== 用户问题向量化测试 ====================
    print("\n" + "=" * 70)
    print("RAG3 - 用户问题向量化测试")
    print("=" * 70)

    test_user_question_vectorization()

    # ==================== 相似度搜索测试 ====================

    # 检查索引是否存在
    try:
        from app.service.core.vector_store import get_vector_search_service

        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(INDEX_NAME)

        if not index_exists:
            print(f"\n⚠ 警告: 索引 '{INDEX_NAME}' 不存在")
            print("请先运行文档处理流程创建索引:")
            print("  python RAG3.py (设置 ENABLE_STORAGE=true)")
        else:
            doc_count = search_service.es_store.get_document_count(INDEX_NAME)
            print(f"\n索引 '{INDEX_NAME}' 存在，包含 {doc_count} 个文档块")

            if doc_count == 0:
                print("\n⚠ 索引为空，请先处理文档")
            else:
                test_similarity_search()

    except Exception as e:
        print(f"\n检查索引时出错: {e}")
        print("请确保 Elasticsearch 服务已启动")

    # ==================== 增强检索演示 ====================

    try:
        from app.service.core.vector_store import get_vector_search_service

        search_service = get_vector_search_service()
        index_exists = search_service.es_store.index_exists(INDEX_NAME)

        if index_exists and search_service.es_store.get_document_count(INDEX_NAME) > 0:
            print(f"\n索引 '{INDEX_NAME}' 存在，开始演示增强检索...")
            print(f"文档块数量: {search_service.es_store.get_document_count(INDEX_NAME)}")

            success_count = 0
            # 使用前3个问题进行演示
            demo_questions = TEST_QUESTIONS[:3]

            for idx, question in enumerate(demo_questions, 1):
                print(f"\n{'=' * 70}")
                print(f"问题 {idx}/{len(demo_questions)}: {question}")
                print("=" * 70)

                result = enhanced_search_with_hybrid_and_rerank(
                    question=question,
                    index_name=INDEX_NAME,
                    top_k=TOP_K,
                    keyword_weight=0.3,
                    vector_weight=0.7,
                    enable_rerank=True,
                    enable_query_rewrite=True,
                    verbose=True
                )

                if result.get("success") and result['results']:
                    print(f"\n✓ 增强检索成功")
                    print(f"  召回块数: {result['total_returned']}")
                    print(f"  最佳匹配文档: {result['results'][0]['document_name']}")
                    print(f"  最佳匹配分数: {result['results'][0]['score']:.4f}")
                    success_count += 1
                else:
                    print(f"\n✗ 增强检索失败: {result.get('error', '未知错误')}")

            # 输出汇总
            print("\n" + "=" * 70)
            print("演示完成汇总")
            print("=" * 70)
            print(f"\n总问题数: {len(demo_questions)}")
            print(f"成功: {success_count}")
            print(f"失败: {len(demo_questions) - success_count}")

        else:
            print(f"\n⚠ 索引 '{INDEX_NAME}' 不存在或为空，跳过增强检索演示")
            print("请先运行文档处理流程创建索引")

    except Exception as e:
        print(f"\n增强检索演示失败: {e}")