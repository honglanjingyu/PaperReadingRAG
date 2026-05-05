# fix_milvus.py
from pymilvus import connections, utility

connections.connect(host="172.20.48.1", port="19530")

collection_name = "rag_documents"

if utility.has_collection(collection_name):
    print(f"删除集合: {collection_name}")
    utility.drop_collection(collection_name)
    print("✓ 已删除")

print("请重新运行 RAG4.py 处理文档")

connections.disconnect("default")