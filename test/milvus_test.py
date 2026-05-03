# simple_test.py
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
import random


def simple_test():
    print("开始测试 Milvus...")

    # 连接
    connections.connect(host="172.20.48.1", port="19530")
    print("✓ 连接成功")

    # 查看版本
    print(f"✓ Milvus 版本: {utility.get_server_version()}")

    # 测试集合名称
    collection_name = "test_collection"

    # 删除已存在的测试集合
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"✓ 删除旧集合: {collection_name}")

    # 创建集合
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=8),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=100)
    ]
    schema = CollectionSchema(fields)
    collection = Collection(collection_name, schema)
    print(f"✓ 创建集合: {collection_name}")

    # 插入数据
    ids = list(range(100))
    vectors = [[random.random() for _ in range(8)] for _ in range(100)]
    texts = [f"文本_{i}" for i in range(100)]
    collection.insert([ids, vectors, texts])
    print(f"✓ 插入 100 条数据")

    # 创建索引
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 10}}
    collection.create_index("vector", index_params)
    print(f"✓ 创建索引")

    # 加载到内存
    collection.load()
    print(f"✓ 加载集合到内存")

    # 等待一下
    import time
    time.sleep(1)

    # 搜索
    query_vector = [[random.random() for _ in range(8)]]
    search_params = {"metric_type": "L2", "params": {"nprobe": 5}}
    results = collection.search(query_vector, "vector", search_params, limit=3)

    print(f"\n✓ 搜索完成，找到 {len(results[0])} 个相似向量:")
    for i, result in enumerate(results[0]):
        print(f"  结果 {i + 1}: ID={result.id}, 距离={result.distance:.6f}, 文本={result.entity.get('text')}")

    # 可选：清理数据
    # utility.drop_collection(collection_name)
    # print(f"✓ 清理测试数据")

    print("\n✓ 所有测试通过！")
    connections.disconnect("default")


if __name__ == "__main__":
    simple_test()