# demo/test_parent_child.py
"""
演示脚本 - 测试父子分块和父子查询功能
使用现有的 PDF 文件进行测试
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def demo_parent_child_chunking_with_pdf(pdf_path: str = None):
    """
    测试1: 使用真实 PDF 文件测试父子分块功能
    """
    print("\n" + "=" * 70)
    print("测试1: 使用 PDF 文件测试父子分块功能")
    print("=" * 70)

    from app.service.core.deepdoc import DocumentParser
    from app.service.core.chunking import ParentChildSplitter

    # 使用提供的 PDF 路径
    if pdf_path is None:
        pdf_path = Path(project_root) / "uploads" / "普通用户.pdf"

    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF 文件不存在: {pdf_path}")
        print("请确保文件已上传到 uploads 目录")
        return None

    print(f"\nPDF 文件: {pdf_path}")

    # 1. 解析 PDF
    print("\n步骤1: 解析 PDF 文档...")
    parser = DocumentParser()
    parsed = parser.parse(
        str(pdf_path),
        from_page=0,
        to_page=100000,
        enable_cleaning=True,
        verbose=False
    )

    if not parsed.cleaned_text:
        print("❌ 解析失败: 未能提取文本内容")
        return None

    print(f"  ✅ 解析成功")
    print(f"  文件名: {parsed.file_name}")
    print(f"  文本长度: {len(parsed.cleaned_text)} 字符")
    print(f"  文本预览: {parsed.cleaned_text[:200]}...")

    # 2. 父子分块
    print("\n步骤2: 父子分块...")
    chunker = ParentChildSplitter(
        parent_chunk_size=500,  # 父块大小（tokens）
        child_chunk_size=150,  # 子块大小（tokens）
        parent_overlap=50,
        child_overlap=20,
        min_child_size=30
    )

    doc = chunker.split_document(
        text=parsed.cleaned_text,
        metadata={
            'source': parsed.file_name,
            'file_type': parsed.file_type,
            'total_pages': parsed.total_pages
        },
        document_id="test_doc_001",
        document_name=parsed.file_name
    )

    # 获取统计信息
    stats = chunker.get_statistics(doc)

    print(f"\n分块统计:")
    print(f"  - 父块数量: {stats['parent_count']}")
    print(f"  - 子块数量: {stats['child_count']}")
    print(f"  - 平均父块大小: {stats['avg_parent_tokens']:.0f} tokens")
    print(f"  - 平均子块大小: {stats['avg_child_tokens']:.0f} tokens")
    print(f"  - 最小父块: {stats['min_parent_tokens']} tokens")
    print(f"  - 最大父块: {stats['max_parent_tokens']} tokens")

    # 显示父块内容
    print(f"\n父块详情:")
    for i, parent in enumerate(doc.parent_chunks):
        print(f"\n  【父块 {i + 1}】 (索引: {parent.chunk_index}, tokens: {parent.token_count})")
        content_preview = parent.content[:200].replace('\n', ' ')
        print(f"    内容预览: {content_preview}...")
        print(f"    包含子块数: {len(parent.children)}")

        # 显示子块
        for j, child in enumerate(parent.children[:2]):
            print(f"      子块{j + 1}: {child.content[:80]}...")

    return doc


def demo_process_and_store(pdf_path: str = None):
    """
    测试2: 完整处理流程 - 解析 -> 父子分块 -> 向量化 -> 存储
    """
    print("\n" + "=" * 70)
    print("测试2: 完整处理流程（解析 -> 分块 -> 向量化 -> 存储）")
    print("=" * 70)

    from app.service.core.rag import process_document_parent_child
    from app.service.core.vector_store import get_vector_store

    # 使用提供的 PDF 路径
    if pdf_path is None:
        pdf_path = Path(project_root) / "uploads" / "普通用户.pdf"

    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF 文件不存在: {pdf_path}")
        print("请确保文件已上传到 uploads 目录")
        return None

    print(f"\nPDF 文件: {pdf_path}")
    print(f"用户等级: normal")

    # 执行完整处理流程
    print("\n开始处理文档...")

    parent_child_doc, vector_chunks = process_document_parent_child(
        file_path=str(pdf_path),
        parent_chunk_size=500,
        child_chunk_size=150,
        enable_vectorization=True,
        enable_storage=True,
        verbose=True,
        user_level="normal"
    )

    if parent_child_doc and vector_chunks:
        print(f"\n✅ 文档处理成功")
        print(f"  父块数: {len(parent_child_doc.parent_chunks)}")
        print(f"  子块数: {len(parent_child_doc.all_children)}")
        print(f"  向量化块数: {len(vector_chunks)}")

        # 验证存储
        index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")
        store = get_vector_store()
        if store.index_exists(index_name):
            doc_count = store.get_document_count(index_name)
            print(f"\n验证存储:")
            print(f"  索引名称: {index_name}")
            print(f"  文档总数: {doc_count}")
        return parent_child_doc, vector_chunks
    else:
        print(f"\n❌ 文档处理失败")
        if not vector_chunks:
            print("  原因: 向量化失败")
        return None, None


def demo_parent_child_retrieval():
    """
    测试3: 父子查询功能
    测试从已存储的文档中检索
    """
    print("\n" + "=" * 70)
    print("测试3: 父子查询功能")
    print("=" * 70)

    from app.service.core.retrieval import get_parent_child_retriever
    from app.service.core.vector_store import get_vector_store

    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    # 检查索引是否存在
    store = get_vector_store()

    if not store.index_exists(index_name):
        print(f"\n⚠️ 索引 '{index_name}' 不存在")
        print("请先运行测试2处理文档")
        return None

    doc_count = store.get_document_count(index_name)
    print(f"\n索引 '{index_name}' 中共有 {doc_count} 条记录")

    if doc_count == 0:
        print("⚠️ 索引为空，请先运行测试2处理文档")
        return None

    # 创建检索器
    retriever = get_parent_child_retriever()
    print(f"\n检索器初始化成功")

    # 基于 PDF 内容的测试查询
    test_queries = [
        "XX科技有限公司成立于哪一年",
        "公司的主要业务有哪些",
        "公司的核心价值观是什么",
        "公司完成了什么融资",
        "公司的合作伙伴有哪些",
    ]

    results = []
    for query in test_queries:
        print(f"\n{'─' * 50}")
        print(f"查询: {query}")

        # 执行检索
        result = retriever.search_with_query_rewrite(
            question=query,
            index_name=index_name,
            top_k=3,
            recall_k=10,
            enable_rerank=False,
            enable_query_rewrite=True
        )

        if result.get("success"):
            retrieved_results = result.get("results", [])
            print(f"  ✅ 检索成功")
            print(f"  召回结果数: {result.get('total_recalled', 0)}")
            print(f"  返回结果数: {len(retrieved_results)}")

            for i, doc_result in enumerate(retrieved_results[:3], 1):
                score = doc_result.get('score', 0)
                content = doc_result.get('content', '')[:150].replace('\n', ' ')
                parent_id = doc_result.get('parent_id', 'N/A')[:20]
                child_count = doc_result.get('child_count', 0)
                print(f"\n  结果 {i}:")
                print(f"    分数: {score:.4f}")
                print(f"    父块ID: {parent_id}...")
                print(f"    包含子块数: {child_count}")
                print(f"    内容: {content}...")

            results.append({"query": query, "success": True, "count": len(retrieved_results)})
        else:
            print(f"  ❌ 检索失败: {result.get('error', '未知错误')}")
            results.append({"query": query, "success": False, "error": result.get('error')})

    # 汇总
    print("\n" + "=" * 50)
    print("检索汇总:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        if r["success"]:
            print(f"  {status} {r['query']}: 找到 {r['count']} 条结果")
        else:
            print(f"  {status} {r['query']}: {r.get('error', '失败')}")

    return results


def demo_compare_without_storage(pdf_path: str = None):
    """
    测试4: 不存储，仅对比分块结果
    """
    print("\n" + "=" * 70)
    print("测试4: 分块结果预览（不存储）")
    print("=" * 70)

    from app.service.core.deepdoc import DocumentParser
    from app.service.core.chunking import ParentChildSplitter, chunk_text_simple

    # 使用提供的 PDF 路径
    if pdf_path is None:
        pdf_path = Path(project_root) / "uploads" / "普通用户.pdf"

    if not os.path.exists(pdf_path):
        print(f"\n❌ PDF 文件不存在: {pdf_path}")
        return None

    print(f"\nPDF 文件: {pdf_path}")

    # 解析 PDF
    print("\n解析 PDF 文档...")
    parser = DocumentParser()
    parsed = parser.parse(
        str(pdf_path),
        from_page=0,
        to_page=100000,
        enable_cleaning=True,
        verbose=False
    )

    if not parsed.cleaned_text:
        print("❌ 解析失败")
        return None

    text = parsed.cleaned_text
    print(f"文本长度: {len(text)} 字符")

    # 传统分块
    print("\n【传统分块】")
    traditional_chunks = chunk_text_simple(text, chunk_size=150)
    print(f"  分块数量: {len(traditional_chunks)}")
    for i, chunk in enumerate(traditional_chunks[:3], 1):
        print(f"  块{i}: {chunk[:100]}...")

    # 父子分块
    print("\n【父子分块】")
    chunker = ParentChildSplitter(
        parent_chunk_size=500,
        child_chunk_size=150,
        parent_overlap=50,
        child_overlap=20
    )
    doc = chunker.split_document(text, document_name=parsed.file_name)

    print(f"  父块数量: {len(doc.parent_chunks)}")
    print(f"  子块数量: {len(doc.all_children)}")

    for i, parent in enumerate(doc.parent_chunks):
        print(f"\n  父块 {i + 1} (tokens: {parent.token_count}):")
        print(f"    内容: {parent.content[:150]}...")
        print(f"    子块数: {len(parent.children)}")
        for j, child in enumerate(parent.children[:2]):
            print(f"      子块{j + 1}: {child.content[:80]}...")

    return doc


def run_all_tests(pdf_path: str = None):
    """
    运行所有测试
    """
    print("\n" + "=" * 70)
    print("父子分块和父子查询功能完整测试")
    print("=" * 70)

    results = {}

    # 测试1: 分块功能
    try:
        doc = demo_parent_child_chunking_with_pdf(pdf_path)
        results['chunking'] = "✅ 通过" if doc else "❌ 失败"
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        results['chunking'] = f"❌ 失败: {e}"

    # 测试2: 完整流程
    try:
        parent_doc, vec_chunks = demo_process_and_store(pdf_path)
        results['process'] = "✅ 通过" if parent_doc and vec_chunks else "❌ 失败"
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        results['process'] = f"❌ 失败: {e}"

    # 测试3: 检索功能
    try:
        retrieval_results = demo_parent_child_retrieval()
        results['retrieval'] = "✅ 通过" if retrieval_results else "❌ 失败"
    except Exception as e:
        print(f"❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        results['retrieval'] = f"❌ 失败: {e}"

    # 测试4: 对比
    try:
        demo_compare_without_storage(pdf_path)
        results['compare'] = "✅ 通过"
    except Exception as e:
        print(f"❌ 测试4失败: {e}")
        results['compare'] = f"❌ 失败: {e}"

    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    for name, result in results.items():
        print(f"  {name}: {result}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    pdf_path = Path(project_root) / "uploads_tmp" / "advanced" / "普通用户.pdf"
    run_all_tests(pdf_path)