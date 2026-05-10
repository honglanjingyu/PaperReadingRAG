# E:\elasticsearch\Dockerfile
# 构建带 IK 分词器和拼音分词器的 Elasticsearch 镜像

FROM elasticsearch:8.11.0

USER root

# 安装 IK 分词器（中文分词）- 使用新的下载源
# 注意：需要替换成你实际的 ES 版本对应的插件版本
RUN bin/elasticsearch-plugin install --batch \
    https://get.infini.cloud/elasticsearch/analysis-ik/8.11.0

# 安装拼音分词器（可选）- 使用新的下载源
RUN bin/elasticsearch-plugin install --batch \
    https://get.infini.cloud/elasticsearch/analysis-pinyin/8.11.0

# 设置中国时区
ENV TZ=Asia/Shanghai

# 切换回 elasticsearch 用户
USER elasticsearch

EXPOSE 9200 9300