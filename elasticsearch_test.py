# fix_and_test.py
from elasticsearch import Elasticsearch
import time

es = Elasticsearch(["http://localhost:9200"], verify_certs=False)

# 1. 彻底删除
print("1. 删除旧索引...")
es.indices.delete(index="rag_documents", ignore_unavailable=True)
time.sleep(1)

# 2. 重新创建
print("2. 创建新索引...")
es.indices.create(index="rag_documents", body={
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "1s"
    },
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "content_with_weight": {"type": "text"},
            "docnm": {"type": "text"},
            "kb_id": {"type": "keyword"},
            "create_time": {"type": "date"},
            "token_count": {"type": "integer"},
            "q_1024_vec": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
})
print("索引创建完成")

# 3. 测试写入
print("\n3. 测试写入...")
test_doc = {
    "id": "test_001",
    "content_with_weight": "这是一个测试文档",
    "docnm": "test.pdf",
    "kb_id": "test_kb",
    "create_time": "2024-01-01T00:00:00",
    "token_count": 10,
    "q_1024_vec": [0.1] * 1024
}

response = es.index(index="rag_documents", id="test_001", document=test_doc)
print(f"写入结果: {response['result']}")

# 4. 强制刷新
print("\n4. 刷新索引...")
es.indices.refresh(index="rag_documents")

# 5. 验证
print("\n5. 验证结果...")
count = es.count(index="rag_documents")
print(f"文档数量: {count['count']}")

if count['count'] > 0:
    print("\n✅ 成功！文档已写入")

    # 搜索验证
    result = es.search(index="rag_documents", body={"query": {"match_all": {}}})
    for hit in result['hits']['hits']:
        print(f"  - ID: {hit['_id']}")
        print(f"    内容: {hit['_source'].get('content_with_weight', 'N/A')[:50]}")
else:
    print("\n❌ 失败！文档未写入")
    print("请检查 ES 日志")