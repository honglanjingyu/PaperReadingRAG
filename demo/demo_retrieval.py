# check_es_docs.py
from app.service.core.retrieval import get_es_bm25_retriever

es = get_es_bm25_retriever()
index_name = "rag_documents"
es_index = f"rag_bm25_{index_name}"

if es.is_available():
    # 检查索引是否存在
    if es._client.indices.exists(index=es_index):
        count = es.get_document_count(index_name)
        print(f"ES 索引 {es_index} 中的文档数: {count}")

        # 获取前几条文档内容
        if count > 0:
            response = es._client.search(index=es_index, size=2)
            for hit in response['hits']['hits']:
                print(f"  - id: {hit['_id']}, source: {hit['_source'].get('docnm')}")
    else:
        print(f"ES 索引 {es_index} 不存在")
else:
    print("ES 不可用")