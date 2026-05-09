# demo_a2a_discovery.py
"""
A2A Agent 发现机制测试脚本
测试 PaperReadingRAG 的 .well-known/agent.json 发现端点
"""

import asyncio
import aiohttp
import json
from datetime import datetime


def demo_print_header(title: str):
    """打印测试标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_print_success(message: str):
    """打印成功信息"""
    print(f"  ✅ {message}")


def demo_print_error(message: str):
    """打印错误信息"""
    print(f"  ❌ {message}")


def demo_print_info(message: str):
    """打印信息"""
    print(f"  📌 {message}")


async def demo_test_well_known_discovery():
    """测试 .well-known 发现端点"""
    demo_print_header("测试 1: .well-known 发现端点")

    base_url = "http://localhost:8001"
    endpoints = [
        "/.well-known/agent.json",
        "/.well-known/agent-card",
    ]

    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            try:
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        demo_print_success(f"{endpoint}")
                        print(f"      Agent名称: {data.get('name', 'N/A')}")
                        print(f"      版本: {data.get('version', 'N/A')}")
                        print(f"      技能数量: {len(data.get('skills', []))}")
                        print(f"      端点数量: {len(data.get('endpoints', {}))}")
                    else:
                        demo_print_error(f"{endpoint}: HTTP {resp.status}")
            except asyncio.TimeoutError:
                demo_print_error(f"{endpoint}: 连接超时")
            except Exception as e:
                demo_print_error(f"{endpoint}: {e}")


async def demo_test_a2a_health():
    """测试 A2A 健康检查"""
    demo_print_header("测试 2: A2A 健康检查")

    base_url = "http://localhost:8001"
    url = f"{base_url}/a2a/health"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    demo_print_success("健康检查通过")
                    print(f"      Agent: {data.get('agent_name', 'N/A')}")
                    print(f"      能力: {', '.join(data.get('capabilities', []))}")
                else:
                    demo_print_error(f"健康检查失败: HTTP {resp.status}")
        except Exception as e:
            demo_print_error(f"健康检查失败: {e}")


async def demo_test_a2a_capabilities():
    """测试 A2A 能力摘要"""
    demo_print_header("测试 3: A2A 能力摘要")

    base_url = "http://localhost:8001"
    endpoints = [
        "/a2a/capabilities",
        "/a2a/info",
    ]

    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            url = f"{base_url}{endpoint}"
            try:
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        demo_print_success(f"{endpoint}")
                        if "agent_name" in data:
                            print(f"      Agent: {data.get('agent_name')}")
                        if "available_skills" in data:
                            print(f"      技能: {', '.join(data.get('available_skills', []))}")
                        if "supports_streaming" in data:
                            print(f"      支持流式: {data.get('supports_streaming')}")
                            print(f"      支持记忆: {data.get('supports_memory')}")
                    else:
                        demo_print_error(f"{endpoint}: HTTP {resp.status}")
            except Exception as e:
                demo_print_error(f"{endpoint}: {e}")


async def demo_test_rag_ask():
    """测试 RAG 问答接口"""
    demo_print_header("测试 4: RAG 问答接口")

    base_url = "http://localhost:8001"
    url = f"{base_url}/a2a/rag/ask"

    test_queries = [
        "什么是RAG技术？",
        "文档的主要内容是什么？",
        "请总结一下",
    ]

    async with aiohttp.ClientSession() as session:
        for query in test_queries:
            data = {
                "query": query,
                "top_k": 3
            }

            try:
                async with session.post(url, json=data, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"\n  📝 问题: {query}")
                        print(f"     成功: {result.get('success')}")
                        print(f"     会话ID: {result.get('session_id', 'N/A')[:16]}...")

                        if result.get('answer'):
                            answer = result['answer'][:150]
                            print(f"     回答: {answer}...")
                        elif result.get('error'):
                            print(f"     错误: {result.get('error')}")
                    else:
                        text = await resp.text()
                        demo_print_error(f"请求失败: HTTP {resp.status}")
            except asyncio.TimeoutError:
                demo_print_error(f"请求超时: {query}")
            except Exception as e:
                demo_print_error(f"请求异常: {e}")

            # 避免请求过快
            await asyncio.sleep(1)


async def demo_test_rag_search():
    """测试 RAG 检索接口"""
    demo_print_header("测试 5: RAG 检索接口")

    base_url = "http://localhost:8001"
    url = f"{base_url}/a2a/rag/search"

    test_queries = [
        "RAG技术",
        "财务报表",
    ]

    async with aiohttp.ClientSession() as session:
        for query in test_queries:
            data = {
                "query": query,
                "need_answer": False,
                "top_k": 3
            }

            try:
                async with session.post(url, json=data, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"\n  🔍 搜索: {query}")
                        print(f"     成功: {result.get('success')}")

                        if result.get('documents'):
                            docs = result['documents']
                            print(f"     找到 {len(docs)} 个相关文档")
                            for i, doc in enumerate(docs[:2], 1):
                                content = doc.get('content', '')[:80]
                                score = doc.get('score', 0)
                                print(f"        [{i}] 相关度: {score:.3f}")
                                print(f"            内容: {content}...")
                        elif result.get('error'):
                            print(f"     错误: {result.get('error')}")
                    else:
                        text = await resp.text()
                        demo_print_error(f"请求失败: HTTP {resp.status}")
            except Exception as e:
                demo_print_error(f"请求异常: {e}")

            await asyncio.sleep(1)


async def demo_test_session_persistence():
    """测试会话持久化"""
    demo_print_header("测试 6: 会话持久化")

    base_url = "http://localhost:8001"
    url = f"{base_url}/a2a/rag/ask"

    session_id = None

    async with aiohttp.ClientSession() as session:
        # 第一轮对话
        print("\n  📝 第一轮对话:")
        data1 = {"query": "你好，请记住我叫张三", "top_k": 3}

        try:
            async with session.post(url, json=data1, timeout=30) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    session_id = result.get('session_id')
                    print(f"     会话ID: {session_id[:16]}...")
                    print(f"     回答: {result.get('answer', '')[:100]}...")
                    demo_print_success("第一轮对话完成")
        except Exception as e:
            demo_print_error(f"第一轮对话失败: {e}")

        # 第二轮对话（使用相同会话）
        if session_id:
            print("\n  📝 第二轮对话（测试记忆）:")
            data2 = {"query": "我叫什么名字？", "session_id": session_id, "top_k": 3}

            try:
                async with session.post(url, json=data2, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"     会话ID: {result.get('session_id')[:16]}...")
                        answer = result.get('answer', '')
                        print(f"     回答: {answer[:150]}...")

                        if "张三" in answer:
                            demo_print_success("会话记忆正常工作")
                        else:
                            demo_print_info("会话记忆可能未生效（可能没有相关文档）")
                    else:
                        demo_print_error(f"第二轮对话失败: HTTP {resp.status}")
            except Exception as e:
                demo_print_error(f"第二轮对话失败: {e}")


async def demo_test_error_handling():
    """测试错误处理"""
    demo_print_header("测试 7: 错误处理")

    base_url = "http://localhost:8001"
    url = f"{base_url}/a2a/rag/ask"

    test_cases = [
        {"query": "", "top_k": 3, "desc": "空查询"},
        {"query": "测试", "top_k": 100, "desc": "过大的 top_k"},
        {"query": "x" * 10000, "top_k": 3, "desc": "超长查询"},
    ]

    async with aiohttp.ClientSession() as session:
        for test in test_cases:
            data = {"query": test["query"], "top_k": test["top_k"]}

            try:
                async with session.post(url, json=data, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"\n  🧪 {test['desc']}")
                        print(f"     成功: {result.get('success')}")
                        if result.get('error'):
                            print(f"     错误信息: {result.get('error')[:100]}")
                        demo_print_success("错误处理正常")
                    else:
                        print(f"\n  🧪 {test['desc']}")
                        demo_print_info(f"返回 HTTP {resp.status}")
            except Exception as e:
                print(f"\n  🧪 {test['desc']}")
                demo_print_error(f"异常: {e}")

            await asyncio.sleep(0.5)


async def demo_get_agent_card():
    """获取并显示完整的 Agent 卡片"""
    demo_print_header("测试 8: Agent 卡片详情")

    base_url = "http://localhost:8001"
    url = f"{base_url}/.well-known/agent.json"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    card = await resp.json()

                    print("\n  📋 Agent 基本信息:")
                    print(f"     ID: {card.get('id', 'N/A')}")
                    print(f"     名称: {card.get('name', 'N/A')}")
                    print(f"     描述: {card.get('description', 'N/A')[:100]}...")
                    print(f"     版本: {card.get('version', 'N/A')}")
                    print(f"     提供者: {card.get('provider', 'N/A')}")

                    print("\n  🎯 技能列表:")
                    for skill in card.get('skills', []):
                        print(f"     - {skill.get('name')}: {skill.get('description')[:60]}...")

                    print("\n  ⚙️ 能力配置:")
                    capabilities = card.get('capabilities', {})
                    for key, value in capabilities.items():
                        print(f"     - {key}: {value}")

                    print("\n  🔗 端点列表:")
                    for name, url_path in card.get('endpoints', {}).items():
                        print(f"     - {name}: {url_path}")

                    demo_print_success("Agent 卡片获取成功")
                else:
                    demo_print_error(f"获取失败: HTTP {resp.status}")
        except Exception as e:
            demo_print_error(f"获取失败: {e}")


async def demo_run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("  A2A Agent 发现机制完整测试")
    print("=" * 70)
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  目标服务: http://localhost:8001")
    print("\n  请确保 PaperReadingRAG 服务已启动:")
    print("    python run_api.py --port 8001")
    print("=" * 70)

    # 等待用户确认
    await asyncio.sleep(2)

    # 运行所有测试
    await demo_test_well_known_discovery()
    await demo_test_a2a_health()
    await demo_test_a2a_capabilities()
    await demo_test_rag_ask()
    await demo_test_rag_search()
    await demo_test_session_persistence()
    await demo_test_error_handling()
    await demo_get_agent_card()

    print("\n" + "=" * 70)
    print("  所有测试完成！")
    print("=" * 70)
    print("\n  Nexus 现在可以通过以下方式发现和使用此 Agent:")
    print("    1. GET /.well-known/agent.json - 获取 Agent 能力")
    print("    2. POST /a2a/rag/ask - 进行问答")
    print("    3. POST /a2a/rag/search - 进行检索")
    print("=" * 70)


async def demo_quick_test():
    """快速测试 - 只测试核心功能"""
    print("\n" + "=" * 70)
    print("  A2A Agent 快速测试")
    print("=" * 70)

    await demo_test_well_known_discovery()
    await demo_test_a2a_health()
    await demo_test_rag_ask()

    print("\n快速测试完成！")


if __name__ == "__main__":
    asyncio.run(demo_run_all_tests())