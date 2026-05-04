# diagnose.py
"""诊断环境变量和连接问题"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

print("=" * 70)
print("环境变量诊断")
print("=" * 70)

print("\n向量存储配置:")
print(f"  VECTOR_STORE_TYPE: {os.getenv('VECTOR_STORE_TYPE', '未设置')}")
print(f"  VECTOR_STORE_HOST: {os.getenv('VECTOR_STORE_HOST', '未设置')}")
print(f"  VECTOR_STORE_PORT: {os.getenv('VECTOR_STORE_PORT', '未设置')}")
print(f"  VECTOR_STORE_USER: {os.getenv('VECTOR_STORE_USER', '未设置')}")
print(f"  VECTOR_STORE_PASSWORD: {'已设置' if os.getenv('VECTOR_STORE_PASSWORD') else '未设置'}")
print(f"  VECTOR_INDEX_NAME: {os.getenv('VECTOR_INDEX_NAME', '未设置')}")

print("\n测试连接:")
try:
    from app.service.core.vector_store import get_vector_store, get_store_type

    store = get_vector_store()
    store_type = get_store_type()

    print(f"  ✓ 存储类型: {store_type}")
    print(f"  ✓ 存储对象: {type(store).__name__}")

    # 测试索引检查
    index_name = os.getenv("VECTOR_INDEX_NAME", "rag_documents")
    exists = store.index_exists(index_name)
    print(f"  ✓ 索引 '{index_name}' 存在: {exists}")

    if exists:
        count = store.get_document_count(index_name)
        print(f"  ✓ 文档数量: {count}")

    print("\n✓ 所有测试通过!")

except Exception as e:
    print(f"  ✗ 测试失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 70)