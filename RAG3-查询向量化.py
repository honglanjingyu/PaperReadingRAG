# RAG2.py - 入口文件，只调用不实现
"""
RAG 文档处理 - 智能分块 + 向量化 + 向量存储

本文件现在只是一个调用入口，所有功能通过导入 app/service/core 模块实现

处理顺序：
1. 数据加载 - 加载文档原始内容（来自 RAG1）
2. 布局识别 - 识别文档布局（来自 RAG1）
3. 连接跨页内容 - 合并跨页段落和表格（来自 RAG1）
4. 数据清洗 - 清洗文本、过滤噪声（来自 RAG1）
5. 智能分块 - 使用 chunking 模块进行分块
6. 向量化 - 使用 embedding 模块生成向量
7. 向量存储 - 使用 vector_store 模块存储到 ES
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
)


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
    RAG2功能：智能分块 -> 向量化 -> 向量存储

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
        print("\nRAG1 流程（文档解析与清洗）:")
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

    if verbose:
        print("\n" + "=" * 70)
        print("处理完成！")
        print("=" * 70)

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
        print("\n" + "=" * 70)
        print("【功能 8 & 9】用户问题向量化")
        print("=" * 70)
        print("\n[8/9] 接收用户问题...")
        print(f"\n用户问题: {question}")
        print(f"问题长度: {len(question)} 字符")

    # 9. 将用户输入的问题向量化
    if verbose:
        print("\n[9/9] 问题向量化...")

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


def test_user_question_vectorization():
    """
    测试用户问题向量化功能
    使用默认测试问题进行测试
    """
    # 默认测试问题
    test_questions = [
        "世运电子的主要业务是什么？",
        "公司2023年中报的营收情况如何？",
        "请分析世运电子的盈利能力",
        "公司的主要客户有哪些？",
        "世运电子的竞争优势是什么？"
    ]

    print("\n" + "=" * 70)
    print("RAG3 - 用户问题向量化测试")
    print("=" * 70)

    # 使用第一个测试问题
    test_question = test_questions[0]

    print(f"\n默认测试问题: {test_question}")

    # 执行向量化
    result = vectorize_user_question(question=test_question, verbose=True)

    # 输出详细结果
    if result and result.get("success"):
        print("\n" + "-" * 70)
        print("向量化结果详情:")
        print("-" * 70)
        print(f"  问题: {result['question']}")
        print(f"  问题长度: {result['question_length']} 字符")
        print(f"  向量维度: {result['vector_dimension']}")
        print(f"  向量前10维: {result['vector'][:10]}")
        print(f"  向量后10维: {result['vector'][-10:]}")
        print(f"  模型类型: {result['model_type']}")
        print(f"  模型名称: {result['model_info']['model_name']}")

        # 可选：显示完整向量（需要时可取消注释）
        # print(f"\n完整向量: {result['vector']}")

        return result
    else:
        print(f"\n向量化失败: {result.get('error', '未知错误')}")
        return None


def process_user_question():
    """
    用户问题处理函数
    接收用户输入问题并输出向量化结果
    """
    print("\n" + "=" * 70)
    print("RAG3 - 用户问题向量化")
    print("=" * 70)

    # 初始化向量化服务
    model_type = os.getenv("EMBEDDING_TYPE", "remote")
    try:
        from app.service.core.embedding import VectorizationService
        vec_service = VectorizationService(model_type)
        model_info = vec_service.get_model_info()
        print(f"\n当前模型配置:")
        print(f"  模型类型: {model_info.get('type', 'unknown')}")
        print(f"  模型名称: {model_info.get('model_name', 'unknown')}")
        print(f"  向量维度: {model_info.get('dimension', 'unknown')}")
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    print("\n" + "-" * 70)

    # 8. 接收用户输入的问题
    question = input("请输入您的问题: ").strip()

    if not question:
        print("错误: 问题不能为空")
        return

    print(f"\n用户问题: {question}")
    print(f"问题长度: {len(question)} 字符")

    # 9. 将用户输入的问题向量化
    print("\n正在生成问题向量...")

    try:
        # 生成向量
        question_vector = vec_service.manager.generate_embedding(question)

        if question_vector is None:
            print("向量化失败")
            return

        print("\n" + "=" * 70)
        print("向量化结果:")
        print("=" * 70)
        print(f"  问题: {question}")
        print(f"  向量维度: {len(question_vector)}")
        print(f"  向量内容: {question_vector}")
        print(f"  向量前10维预览: {question_vector[:10]}...")

    except Exception as e:
        print(f"向量化失败: {e}")

# ==================== 演示 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG2 - 智能分块 + 向量化 + 向量存储（调用入口）")
    print("=" * 70)
    print("\n处理流程:")
    print("  RAG1 流程: 数据加载 -> 布局识别 -> 连接跨页内容 -> 数据清洗")
    print("  RAG2 功能: 智能分块 -> 向量化 -> 向量存储")
    print("=" * 70)

    # 从环境变量读取配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
    ENABLE_VECTORIZATION = os.getenv("ENABLE_VECTORIZATION", "true").lower() == "true"
    ENABLE_STORAGE = os.getenv("ENABLE_STORAGE", "false").lower() == "true"
    MODEL_TYPE = os.getenv("EMBEDDING_TYPE", "local")

    print(f"\n配置信息:")
    print(f"  分块大小: {CHUNK_SIZE} tokens")
    print(f"  向量化: {'启用' if ENABLE_VECTORIZATION else '禁用'}")
    print(f"  存储: {'启用' if ENABLE_STORAGE else '禁用'}")
    if ENABLE_VECTORIZATION:
        print(f"  模型类型: {MODEL_TYPE}")

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

    # ==================== 新增：用户问题向量化测试 ====================

    # 使用默认测试问题进行向量化测试
    test_result = test_user_question_vectorization()

    # 测试多个问题
    print("\n" + "-" * 70)
    print("测试多个问题示例:")
    print("-" * 70)

    test_questions = [
        "世运电子的主要业务是什么？",
        "公司2023年中报的营收情况如何？",
        "请分析世运电子的盈利能力"
    ]

    for i, q in enumerate(test_questions, 1):
        print(f"\n{i}. 问题: {q}")
        result_vec = vectorize_user_question(question=q, verbose=False)
        if result_vec and result_vec.get("success"):
            print(f"   向量维度: {result_vec['vector_dimension']}")
            print(f"   向量前3维: {result_vec['vector'][:3]}...")
        else:
            print(f"   向量化失败")