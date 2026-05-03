# drop_milvus_collection.py
"""删除 Milvus 中的旧集合"""

from pymilvus import connections, utility

# 连接 Milvus
connections.connect(
    alias="default",
    host="172.20.48.1",
    port="19530"
)

collection_name = "rag_documents"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"✓ 集合 '{collection_name}' 已删除")
else:
    print(f"集合 '{collection_name}' 不存在")

connections.disconnect("default")
print("完成")