# 数据获取
# 文档解析
mineru 的 vlm
批量上传异步并发
# 分块
固定 token 数
语义分块
递归分块
句子级分块
段落级分块
overlap
# 向量化
不同embedding模型对比
# 向量化存储
批量上传异步并发
es->milvus 效果提升: 还需要大数据量测试

# Query改写
## 同义词改写

## 同义表维护
TODO: 添加自进化(可以自我维护)
# 相似度搜索
## 向量检索
### mivlus向量索引
- HNSW     需要性能
- IVF_FLAT 目前代码中选择 ANN
- IVF_PQ
- GPU_CAGRA
- DISIKANN
- FLAT

## BM25
rank_bm25->es_bm25 效果提升:20s->50ms
# Rerank
Rerank LLM模型 or Cross-Encoder
# Redis
## 删掉PDF后清理对话
把引用该文档的Agent回答与对应的用户问题一起删掉

# GraphGAG
需要优化neo4j初始建造知识图谱
