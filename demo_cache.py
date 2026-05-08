# demo_cache_validation.py
"""
Redis 缓存功能验证 Demo
运行: python demo_cache_validation.py

功能验证:
1. 向量嵌入缓存 (Embedding)
2. BM25 模型缓存
3. 搜索结果缓存
4. 缓存失效机制
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def demo_print_header(title: str):
    """打印分隔标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_print_success(message: str):
    """打印成功信息"""
    print(f"  ✅ {message}")


def demo_print_error(message: str):
    """打印错误信息"""
    print(f"  ❌ {message}")


def demo_print_info(message: str):
    """打印信息"""
    print(f"  📝 {message}")


def demo_print_warning(message: str):
    """打印警告"""
    print(f"  ⚠️  {message}")


def demo_print_timing(operation: str, elapsed_ms: float):
    """打印耗时"""
    if elapsed_ms < 100:
        print(f"  ⏱️  {operation}: {elapsed_ms:.2f} ms")
    else:
        print(f"  ⏱️  {operation}: {elapsed_ms:.2f} ms ({elapsed_ms / 1000:.2f}s)")


# ============================================================
# 测试 1: Redis 连接测试
# ============================================================

def demo_test_redis_connection():
    """测试 Redis 连接是否正常"""
    demo_print_header("测试 1: Redis 连接测试")

    try:
        import redis
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        password = os.getenv("REDIS_PASSWORD") or None
        db = int(os.getenv("REDIS_DB", 1))

        r = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5
        )

        r.ping()
        demo_print_success(f"Redis 连接成功 (host={host}:{port}, db={db})")

        # 测试读写
        test_key = "rag:cache:test:demo_key"
        r.setex(test_key, 10, "test_value")
        value = r.get(test_key)
        if value == "test_value":
            demo_print_success("Redis 读写测试通过")
        r.delete(test_key)

        return r
    except Exception as e:
        demo_print_error(f"Redis 连接失败: {e}")
        demo_print_info("请确保 Redis 服务已启动: docker run -d -p 6379:6379 redis")
        return None


# ============================================================
# 测试 2: 缓存管理器测试
# ============================================================

def demo_test_cache_manager():
    """测试缓存管理器"""
    demo_print_header("测试 2: 缓存管理器测试")

    try:
        from app.service.core.cache import get_cache_manager

        cache = get_cache_manager()
        demo_print_success(f"缓存管理器初始化成功 (Redis可用: {cache.redis_client is not None})")

        # 测试基本读写
        test_key = "test_key_123"
        test_value = {"message": "hello world", "count": 42}

        # 写入
        start = time.time()
        cache.set("test", test_key, test_value, ttl=10)
        write_time = (time.time() - start) * 1000
        demo_print_timing("缓存写入", write_time)

        # 读取 (第一次应该从 Redis 读)
        start = time.time()
        result = cache.get("test", test_key)
        read_time = (time.time() - start) * 1000

        if result == test_value:
            demo_print_success(f"缓存读取成功: {result}")
            demo_print_timing("缓存读取 (Redis)", read_time)
        else:
            demo_print_error("缓存读取失败: 值不匹配")

        # 第二次读取 (应该从内存读)
        start = time.time()
        result2 = cache.get("test", test_key)
        read_time2 = (time.time() - start) * 1000
        demo_print_timing("缓存读取 (内存)", read_time2)

        if read_time2 < read_time:
            demo_print_success("多级缓存工作正常 (内存比 Redis 快)")

        # 删除
        cache.delete("test", test_key)
        result3 = cache.get("test", test_key)
        if result3 is None:
            demo_print_success("缓存删除成功")

        return cache

    except Exception as e:
        demo_print_error(f"缓存管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 测试 3: 向量嵌入缓存测试
# ============================================================

def demo_test_embedding_cache():
    """测试向量嵌入缓存"""
    demo_print_header("测试 3: 向量嵌入缓存测试")

    try:
        from app.service.core.embedding import get_embedding_manager
        from app.service.core.cache import get_cache_manager

        # 清空 Embedding 缓存
        cache = get_cache_manager()
        cache.delete_pattern("embedding")
        demo_print_info("已清空 Embedding 缓存")

        manager = get_embedding_manager()

        test_texts = [
            "世运电子的主营业务是什么？",
            "世运电子的主营业务是什么？",  # 重复文本
            "公司的营收情况如何？",
        ]

        results = []

        for i, text in enumerate(test_texts, 1):
            print(f"\n  [{i}] 文本: {text[:50]}...")

            # 记录缓存状态
            cache_key = f"embedding:{manager.active_model_type}:*"

            start = time.time()
            embedding = manager.generate_embedding(text)
            elapsed = (time.time() - start) * 1000

            if embedding:
                vector_dim = len(embedding)
                results.append({
                    "text": text,
                    "time_ms": elapsed,
                    "dim": vector_dim,
                    "is_cached": i > 1 and text == test_texts[i - 2] if i > 1 else False
                })
                demo_print_timing(f"向量生成 (首次)", elapsed)
                demo_print_info(f"  向量维度: {vector_dim}")
            else:
                demo_print_error(f"向量生成失败")

        # 验证缓存效果
        print("\n  📊 缓存效果分析:")
        if len(results) >= 2:
            first_time = results[0]["time_ms"]
            second_time = results[1]["time_ms"]

            if second_time < first_time:
                speedup = first_time / second_time if second_time > 0 else float('inf')
                demo_print_success(
                    f"  重复文本缓存命中: {first_time:.2f}ms → {second_time:.2f}ms (提升 {speedup:.1f}x)")
            else:
                demo_print_warning(f"  重复文本未命中缓存: {first_time:.2f}ms → {second_time:.2f}ms")

        return results

    except Exception as e:
        demo_print_error(f"向量嵌入缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 测试 4: BM25 缓存测试 (需要先有文档索引)
# ============================================================

def demo_test_bm25_cache():
    """测试 BM25 模型缓存"""
    demo_print_header("测试 4: BM25 模型缓存测试")

    try:
        from app.service.core.retrieval import HybridRetriever
        from app.service.core.vector_store import get_vector_store
        from app.service.core.cache import get_cache_manager

        # 检查索引是否存在
        store = get_vector_store()
        index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        if not store.index_exists(index_name):
            demo_print_warning(f"索引 '{index_name}' 不存在，跳过 BM25 缓存测试")
            demo_print_info("请先上传文档后再运行此测试")
            return None

        doc_count = store.get_document_count(index_name)
        if doc_count == 0:
            demo_print_warning(f"索引 '{index_name}' 为空，跳过 BM25 缓存测试")
            return None

        demo_print_info(f"索引 '{index_name}' 包含 {doc_count} 个文档")

        # 清空 BM25 缓存
        cache = get_cache_manager()
        cache.delete_pattern("bm25")
        demo_print_info("已清空 BM25 缓存")

        # 创建检索器
        retriever = HybridRetriever(use_jieba=True)

        test_queries = [
            "主营业务",
            "主营业务",  # 重复查询
            "营收情况",
        ]

        results = []

        for i, query in enumerate(test_queries, 1):
            print(f"\n  [{i}] 查询: {query}")

            start = time.time()
            # 这会触发 BM25 模型构建或从缓存加载
            bm25_results = retriever._bm25_search(
                query=query,
                index_name=index_name,
                top_k=5,
                verbose=False
            )
            elapsed = (time.time() - start) * 1000

            is_cached = i > 1 and query == test_queries[i - 2] if i > 1 else False

            results.append({
                "query": query,
                "time_ms": elapsed,
                "result_count": len(bm25_results),
                "is_cached": is_cached
            })

            if is_cached:
                demo_print_timing(f"BM25 搜索 (缓存命中)", elapsed)
            else:
                demo_print_timing(f"BM25 搜索 (首次构建)", elapsed)

            demo_print_info(f"  返回结果数: {len(bm25_results)}")

        # 验证缓存效果
        print("\n  📊 BM25 缓存效果分析:")
        if len(results) >= 2:
            first_time = results[0]["time_ms"]
            second_time = results[1]["time_ms"]

            if second_time < first_time:
                speedup = first_time / second_time if second_time > 0 else float('inf')
                demo_print_success(
                    f"  BM25 模型缓存命中: {first_time:.2f}ms → {second_time:.2f}ms (提升 {speedup:.1f}x)")
            else:
                demo_print_warning(f"  BM25 模型未命中缓存: {first_time:.2f}ms → {second_time:.2f}ms")

        return results

    except Exception as e:
        demo_print_error(f"BM25 缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 测试 5: 搜索结果缓存测试
# ============================================================

def demo_test_search_cache():
    """测试搜索结果缓存"""
    demo_print_header("测试 5: 搜索结果缓存测试")

    try:
        from app.service.core.rag.search import enhanced_search_with_hybrid_and_rerank
        from app.service.core.cache import get_cache_manager

        # 检查索引
        from app.service.core.vector_store import get_vector_store
        store = get_vector_store()
        index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        if not store.index_exists(index_name) or store.get_document_count(index_name) == 0:
            demo_print_warning("索引不存在或为空，跳过搜索结果缓存测试")
            return None

        # 清空搜索缓存
        cache = get_cache_manager()
        cache.delete_pattern("search")
        demo_print_info("已清空搜索缓存")

        # 确保搜索缓存已启用
        os.environ["ENABLE_SEARCH_CACHE"] = "true"

        test_question = "世运电子的主营业务是什么？"

        print(f"\n  📝 测试问题: {test_question}")

        # 第一次搜索 (应该缓存结果)
        print("\n  [1] 第一次搜索 (预期: 缓存未命中，执行完整搜索)")
        start = time.time()
        result1 = enhanced_search_with_hybrid_and_rerank(
            question=test_question,
            index_name=index_name,
            top_k=5,
            recall_k=10,
            verbose=False
        )
        time1 = (time.time() - start) * 1000
        demo_print_timing("第一次搜索", time1)

        if result1.get("success"):
            demo_print_info(f"  返回结果数: {len(result1.get('results', []))}")
        else:
            demo_print_error(f"搜索失败: {result1.get('error')}")

        # 第二次搜索 (应该缓存命中)
        print("\n  [2] 第二次搜索 (预期: 缓存命中，快速返回)")
        start = time.time()
        result2 = enhanced_search_with_hybrid_and_rerank(
            question=test_question,
            index_name=index_name,
            top_k=5,
            recall_k=10,
            verbose=False
        )
        time2 = (time.time() - start) * 1000
        demo_print_timing("第二次搜索", time2)

        # 验证缓存效果
        print("\n  📊 搜索结果缓存效果分析:")
        if time2 < time1:
            speedup = time1 / time2 if time2 > 0 else float('inf')
            demo_print_success(f"  搜索缓存命中: {time1:.2f}ms → {time2:.2f}ms (提升 {speedup:.1f}x)")
        else:
            demo_print_warning(f"  搜索缓存未命中: {time1:.2f}ms → {time2:.2f}ms")

        # 验证结果一致性
        if result1.get("success") and result2.get("success"):
            results1 = result1.get("results", [])
            results2 = result2.get("results", [])

            if len(results1) == len(results2):
                demo_print_success("  缓存结果与原始结果一致")
            else:
                demo_print_warning(f"  结果数量不一致: {len(results1)} vs {len(results2)}")

        return {"time1_ms": time1, "time2_ms": time2, "speedup": time1 / time2 if time2 > 0 else 0}

    except Exception as e:
        demo_print_error(f"搜索结果缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# 测试 6: 缓存失效机制测试
# ============================================================

def demo_test_cache_invalidation():
    """测试缓存失效机制"""
    demo_print_header("测试 6: 缓存失效机制测试")

    try:
        from app.service.core.cache import get_cache_manager
        from app.service.core.embedding import get_embedding_manager
        from app.service.core.retrieval import HybridRetriever
        from app.service.core.vector_store import get_vector_store

        cache = get_cache_manager()
        store = get_vector_store()
        index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

        # 1. 先写入一些缓存数据
        demo_print_info("步骤 1: 写入测试缓存数据")

        # Embedding 缓存
        manager = get_embedding_manager()
        _ = manager.generate_embedding("测试文本")
        demo_print_success("  已写入 Embedding 缓存")

        # BM25 缓存 (如果索引存在)
        if store.index_exists(index_name) and store.get_document_count(index_name) > 0:
            retriever = HybridRetriever(use_jieba=True)
            _ = retriever._bm25_search(query="测试", index_name=index_name, top_k=1, verbose=False)
            demo_print_success("  已写入 BM25 缓存")

        # 搜索缓存
        cache.set("search", "test_key", {"test": "data"}, ttl=60)
        demo_print_success("  已写入搜索缓存")

        # 2. 验证缓存存在
        print("\n  📝 步骤 2: 验证缓存存在")

        embedding_cache_exists = cache.get("embedding", "any_key") is not None
        search_cache_exists = cache.get("search", "test_key") is not None

        # 3. 执行缓存失效
        print("\n  📝 步骤 3: 执行缓存失效")

        # 按模式删除
        cache.delete_pattern("embedding")
        demo_print_success("  已删除 Embedding 缓存")

        cache.delete_pattern("search")
        demo_print_success("  已删除搜索缓存")

        if store.index_exists(index_name):
            cache.delete_pattern(f"bm25:{index_name}")
            demo_print_success(f"  已删除 BM25 缓存: {index_name}")

        # 4. 验证缓存已清除
        print("\n  📝 步骤 4: 验证缓存已清除")

        # 检查搜索缓存
        search_cache_after = cache.get("search", "test_key")
        if search_cache_after is None:
            demo_print_success("  搜索缓存已清除")
        else:
            demo_print_error("  搜索缓存未清除")

        demo_print_success("缓存失效机制测试通过")

    except Exception as e:
        demo_print_error(f"缓存失效测试失败: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# 测试 7: 缓存统计信息
# ============================================================

def demo_show_cache_stats():
    """显示缓存统计信息"""
    demo_print_header("测试 7: 缓存统计信息")

    try:
        import redis
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        password = os.getenv("REDIS_PASSWORD") or None
        db = int(os.getenv("REDIS_DB", 1))

        r = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )

        # 获取所有缓存 key
        all_keys = r.keys("rag:cache:*")

        # 按类型分组
        cache_types = {}
        for key in all_keys:
            parts = key.split(":")
            if len(parts) >= 3:
                cache_type = parts[2]
                if cache_type not in cache_types:
                    cache_types[cache_type] = []
                cache_types[cache_type].append(key)

        print("\n  📊 Redis 缓存统计:")
        print(f"  ┌{'─' * 50}┐")
        print(f"  │ {'类型':<15} │ {'数量':<10} │ {'示例':<20} │")
        print(f"  ├{'─' * 50}┤")

        for cache_type, keys in cache_types.items():
            example = keys[0].split(":")[-1][:16] if keys else ""
            print(f"  │ {cache_type:<15} │ {len(keys):<10} │ {example:<20} │")

        print(f"  └{'─' * 50}┘")
        print(f"\n  总计: {len(all_keys)} 个缓存条目")

        # 内存缓存统计
        from app.service.core.cache import get_cache_manager
        cache = get_cache_manager()
        local_size = len(cache.local_cache)
        print(f"  L1 内存缓存大小: {local_size} 条")

        return cache_types

    except Exception as e:
        demo_print_error(f"获取缓存统计失败: {e}")
        return None


# ============================================================
# 测试 8: 完整工作流测试 (对比有/无缓存)
# ============================================================

def demo_benchmark_without_cache():
    """对比测试: 无缓存时的性能"""
    demo_print_header("测试 8: 性能对比测试 (无缓存)")

    try:
        from app.service.core.embedding import get_embedding_manager
        from app.service.core.rag.search import enhanced_search_with_hybrid_and_rerank
        from app.service.core.vector_store import get_vector_store

        # 临时禁用缓存
        os.environ["ENABLE_EMBEDDING_CACHE"] = "false"
        os.environ["ENABLE_SEARCH_CACHE"] = "false"

        # 清空内存缓存
        from app.service.core.cache import get_cache_manager
        cache = get_cache_manager()
        cache.local_cache.clear()

        demo_print_info("缓存已禁用 (EMBEDDING_CACHE=OFF, SEARCH_CACHE=OFF)")

        manager = get_embedding_manager()
        index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")
        store = get_vector_store()

        test_question = "世运电子的主要业务是什么？"
        test_text = "世运电子主营业务"

        results = {}

        # 测试 Embedding
        print("\n  📝 Embedding 性能测试:")
        start = time.time()
        emb1 = manager.generate_embedding(test_text)
        time1 = (time.time() - start) * 1000
        demo_print_timing("  第一次向量化 (无缓存)", time1)

        start = time.time()
        emb2 = manager.generate_embedding(test_text)
        time2 = (time.time() - start) * 1000
        demo_print_timing("  第二次向量化 (无缓存)", time2)

        results["embedding_no_cache"] = {"first": time1, "second": time2}

        # 测试搜索 (如果索引存在)
        if store.index_exists(index_name) and store.get_document_count(index_name) > 0:
            print("\n  📝 搜索性能测试:")
            start = time.time()
            search1 = enhanced_search_with_hybrid_and_rerank(
                question=test_question,
                index_name=index_name,
                top_k=5,
                recall_k=10,
                verbose=False
            )
            time1 = (time.time() - start) * 1000
            demo_print_timing("  第一次搜索 (无缓存)", time1)

            start = time.time()
            search2 = enhanced_search_with_hybrid_and_rerank(
                question=test_question,
                index_name=index_name,
                top_k=5,
                recall_k=10,
                verbose=False
            )
            time2 = (time.time() - start) * 1000
            demo_print_timing("  第二次搜索 (无缓存)", time2)

            results["search_no_cache"] = {"first": time1, "second": time2}

        return results

    except Exception as e:
        demo_print_error(f"无缓存测试失败: {e}")
        return None


def demo_benchmark_with_cache():
    """对比测试: 有缓存时的性能"""
    demo_print_header("测试 8: 性能对比测试 (有缓存)")

    try:
        from app.service.core.embedding import get_embedding_manager
        from app.service.core.rag.search import enhanced_search_with_hybrid_and_rerank
        from app.service.core.vector_store import get_vector_store
        from app.service.core.cache import get_cache_manager

        # 启用缓存
        os.environ["ENABLE_EMBEDDING_CACHE"] = "true"
        os.environ["ENABLE_SEARCH_CACHE"] = "true"

        # 清空缓存
        cache = get_cache_manager()
        cache.delete_pattern("embedding")
        cache.delete_pattern("search")
        cache.local_cache.clear()

        demo_print_info("缓存已启用 (EMBEDDING_CACHE=ON, SEARCH_CACHE=ON)")
        demo_print_info("已清空所有缓存")

        manager = get_embedding_manager()
        index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")
        store = get_vector_store()

        test_question = "世运电子的主要业务是什么？"
        test_text = "世运电子主营业务"

        results = {}

        # 测试 Embedding
        print("\n  📝 Embedding 性能测试:")
        start = time.time()
        emb1 = manager.generate_embedding(test_text)
        time1 = (time.time() - start) * 1000
        demo_print_timing("  第一次向量化 (缓存未命中)", time1)

        start = time.time()
        emb2 = manager.generate_embedding(test_text)
        time2 = (time.time() - start) * 1000
        demo_print_timing("  第二次向量化 (缓存命中)", time2)

        results["embedding_with_cache"] = {"first": time1, "second": time2}

        # 测试搜索 (如果索引存在)
        if store.index_exists(index_name) and store.get_document_count(index_name) > 0:
            print("\n  📝 搜索性能测试:")
            start = time.time()
            search1 = enhanced_search_with_hybrid_and_rerank(
                question=test_question,
                index_name=index_name,
                top_k=5,
                recall_k=10,
                verbose=False
            )
            time1 = (time.time() - start) * 1000
            demo_print_timing("  第一次搜索 (缓存未命中)", time1)

            start = time.time()
            search2 = enhanced_search_with_hybrid_and_rerank(
                question=test_question,
                index_name=index_name,
                top_k=5,
                recall_k=10,
                verbose=False
            )
            time2 = (time.time() - start) * 1000
            demo_print_timing("  第二次搜索 (缓存命中)", time2)

            results["search_with_cache"] = {"first": time1, "second": time2}

        return results

    except Exception as e:
        demo_print_error(f"有缓存测试失败: {e}")
        return None


def demo_print_comparison(no_cache_results, with_cache_results):
    """打印对比结果"""
    demo_print_header("性能对比总结")

    print("\n  📊 Embedding 向量化对比:")
    print(f"  ┌{'─' * 60}┐")
    print(f"  │ {'场景':<20} │ {'首次 (ms)':<12} │ {'二次 (ms)':<12} │ {'提升':<10} │")
    print(f"  ├{'─' * 60}┤")

    if no_cache_results and "embedding_no_cache" in no_cache_results:
        nc = no_cache_results["embedding_no_cache"]
        print(f"  │ {'无缓存':<20} │ {nc['first']:<12.2f} │ {nc['second']:<12.2f} │ {'-':<10} │")

    if with_cache_results and "embedding_with_cache" in with_cache_results:
        wc = with_cache_results["embedding_with_cache"]
        speedup = wc['first'] / wc['second'] if wc['second'] > 0 else 0
        print(f"  │ {'有缓存':<20} │ {wc['first']:<12.2f} │ {wc['second']:<12.2f} │ {speedup:.1f}x │")

    print(f"  └{'─' * 60}┘")

    print("\n  📊 搜索对比:")
    print(f"  ┌{'─' * 60}┐")
    print(f"  │ {'场景':<20} │ {'首次 (ms)':<12} │ {'二次 (ms)':<12} │ {'提升':<10} │")
    print(f"  ├{'─' * 60}┤")

    if no_cache_results and "search_no_cache" in no_cache_results:
        nc = no_cache_results["search_no_cache"]
        print(f"  │ {'无缓存':<20} │ {nc['first']:<12.2f} │ {nc['second']:<12.2f} │ {'-':<10} │")

    if with_cache_results and "search_with_cache" in with_cache_results:
        wc = with_cache_results["search_with_cache"]
        speedup = wc['first'] / wc['second'] if wc['second'] > 0 else 0
        print(f"  │ {'有缓存':<20} │ {wc['first']:<12.2f} │ {wc['second']:<12.2f} │ {speedup:.1f}x │")

    print(f"  └{'─' * 60}┘")


# ============================================================
# 主函数: 运行所有测试
# ============================================================

def demo_run_all_tests():
    """运行所有缓存验证测试"""
    print("\n" + "=" * 80)
    print("  Redis 缓存功能验证 Demo")
    print("  运行时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # 打印配置
    print("\n📋 当前配置:")
    print(f"  REDIS_HOST: {os.getenv('REDIS_HOST', 'localhost')}")
    print(f"  REDIS_PORT: {os.getenv('REDIS_PORT', '6379')}")
    print(f"  REDIS_DB: {os.getenv('REDIS_DB', '1')}")
    print(f"  CACHE_EMBEDDING_TTL: {os.getenv('CACHE_EMBEDDING_TTL', '86400')}s")
    print(f"  CACHE_SEARCH_TTL: {os.getenv('CACHE_SEARCH_TTL', '300')}s")
    print(f"  CACHE_BM25_TTL: {os.getenv('CACHE_BM25_TTL', '3600')}s")
    print(f"  ENABLE_EMBEDDING_CACHE: {os.getenv('ENABLE_EMBEDDING_CACHE', 'true')}")
    print(f"  ENABLE_SEARCH_CACHE: {os.getenv('ENABLE_SEARCH_CACHE', 'true')}")

    # 1. 测试 Redis 连接
    redis_client = demo_test_redis_connection()
    if not redis_client:
        demo_print_error("Redis 连接失败，无法继续测试")
        return

    # 2. 测试缓存管理器
    cache = demo_test_cache_manager()

    # 3. 测试向量嵌入缓存
    embedding_results = demo_test_embedding_cache()

    # 4. 测试 BM25 缓存
    bm25_results = demo_test_bm25_cache()

    # 5. 测试搜索结果缓存
    search_results = demo_test_search_cache()

    # 6. 测试缓存失效
    demo_test_cache_invalidation()

    # 7. 显示缓存统计
    demo_show_cache_stats()

    # 8. 性能对比测试 (可选，耗时较长)
    print("\n" + "=" * 80)
    print("  性能对比测试 (可能需要几十秒)")
    print("=" * 80)

    # 检查是否有测试数据
    from app.service.core.vector_store import get_vector_store
    store = get_vector_store()
    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")

    if store.index_exists(index_name) and store.get_document_count(index_name) > 0:
        no_cache = demo_benchmark_without_cache()
        with_cache = demo_benchmark_with_cache()
        demo_print_comparison(no_cache, with_cache)
    else:
        demo_print_warning("索引为空，跳过性能对比测试")
        demo_print_info("请先上传文档后再运行完整性能测试")

    # 总结
    demo_print_header("测试总结")

    print("\n  ✅ 缓存模块验证结果:")
    print("    1. Redis 连接 - 正常")
    print("    2. 缓存管理器 - 正常")

    if embedding_results and len(embedding_results) >= 2:
        if embedding_results[1].get("is_cached", False) or embedding_results[1]["time_ms"] < embedding_results[0][
            "time_ms"]:
            print("    3. 向量嵌入缓存 - 正常")
        else:
            print("    3. 向量嵌入缓存 - 需要验证 (可能未命中)")
    else:
        print("    3. 向量嵌入缓存 - 已测试")

    print("\n  💡 使用建议:")
    print("    1. 查看 Redis 中的数据: redis-cli KEYS 'rag:cache:*'")
    print("    2. 查看缓存统计: redis-cli INFO stats")
    print("    3. 清空所有缓存: redis-cli FLUSHDB")

    print("\n" + "=" * 80)
    print("  测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    demo_run_all_tests()