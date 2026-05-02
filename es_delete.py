from app.service.core.vector_store import ESVectorStore
store = ESVectorStore()
if store.index_exists('rag_documents'):
    store.delete_index('rag_documents')
    print('索引已删除')
else:
    print('索引不存在')