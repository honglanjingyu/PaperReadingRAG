# app/service/core/graphrag/community_detection.py
"""
社区发现模块 - 使用 Leiden 算法识别实体社区（纯 igraph 实现，高性能）
Leiden 算法优势：
- 保证社区之间是连通的
- 更好的社区质量
- 更快的收敛速度
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 导入 igraph 和 leidenalg（必需）
try:
    import igraph as ig
    from leidenalg import find_partition, ModularityVertexPartition
    LEIDEN_AVAILABLE = True
    logger.info("Leiden 算法可用 (igraph + leidenalg)")
except ImportError as e:
    LEIDEN_AVAILABLE = False
    logger.error(f"Leiden 算法不可用: {e}")
    logger.error("请安装依赖: pip install igraph leidenalg")
    raise ImportError("Leiden 算法是必需的，请运行: pip install igraph leidenalg")


@dataclass
class Community:
    """社区数据结构"""
    id: int
    entities: List[str]
    size: int
    density: float = 0.0
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    level: int = 0
    modularity: float = 0.0


class CommunityDetector:
    """
    社区发现器 - 使用 Leiden 算法（纯 igraph 实现，高性能）
    """

    def __init__(self, min_community_size: int = 2, resolution: float = 1.0):
        """
        初始化社区检测器

        Args:
            min_community_size: 最小社区大小（少于该值的社区将被过滤）
            resolution: 分辨率参数，越高社区越多
        """
        if not LEIDEN_AVAILABLE:
            raise RuntimeError("Leiden 算法不可用，请安装 igraph 和 leidenalg")

        self.min_community_size = min_community_size
        self.resolution = resolution
        self.communities: Dict[int, Community] = {}

        logger.info(
            f"CommunityDetector 初始化完成, 算法=Leiden, min_community_size={min_community_size}, resolution={resolution}"
        )

    def detect_communities(
            self,
            entities: List[Dict],
            relations: List[Dict]
    ) -> Dict[int, Community]:
        """
        检测实体社区（使用 Leiden 算法）

        Args:
            entities: 实体列表，格式 [{"name": str, "type": str, "frequency": int}, ...]
            relations: 关系列表，格式 [{"source": str, "target": str, "weight": float}, ...]

        Returns:
            社区字典 {community_id: Community}
        """
        if not entities or not relations:
            logger.warning("实体或关系为空，无法检测社区")
            return {}

        # 构建 igraph
        ig_graph, node_names = self._build_igraph(entities, relations)

        if ig_graph.vcount() < 2:
            logger.warning("节点数不足，无法检测社区")
            return {}

        if ig_graph.ecount() == 0:
            logger.warning("图中没有边，无法检测社区")
            return {}

        # 使用 Leiden 算法进行社区检测
        partition = self._leiden_partition(ig_graph)

        # 构建社区对象
        self.communities = self._build_communities(ig_graph, partition, node_names)

        # 过滤小社区
        if self.min_community_size > 1:
            self.communities = {
                cid: c for cid, c in self.communities.items()
                if c.size >= self.min_community_size
            }

        # 计算社区摘要
        self._compute_community_summaries()

        logger.info(f"Leiden 社区检测完成: {len(self.communities)} 个社区")
        return self.communities

    def _build_igraph(self, entities: List[Dict], relations: List[Dict]) -> Tuple[ig.Graph, List[str]]:
        """
        构建 igraph 图（高性能）

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            (igraph图, 节点名称列表)
        """
        # 创建节点名称到索引的映射
        node_names = [e["name"] for e in entities]
        node_index = {name: i for i, name in enumerate(node_names)}

        # 构建边列表
        edges = []
        weights = []
        edge_types = []

        for relation in relations:
            source = relation.get("source")
            target = relation.get("target")
            weight = relation.get("weight", 1.0)
            rel_type = relation.get("relation_type", "RELATED_TO")

            if source in node_index and target in node_index:
                edges.append((node_index[source], node_index[target]))
                weights.append(weight)
                edge_types.append(rel_type)

        # 创建 igraph
        g = ig.Graph()
        g.add_vertices(len(node_names))

        if edges:
            g.add_edges(edges)
            g.es['weight'] = weights
            g.es['relation_type'] = edge_types

        # 添加节点属性
        for i, entity in enumerate(entities):
            g.vs[i]['name'] = entity['name']
            g.vs[i]['type'] = entity.get('type', 'UNKNOWN')
            g.vs[i]['frequency'] = entity.get('frequency', 1)

        logger.info(f"igraph 构建完成: {g.vcount()} 个节点, {g.ecount()} 条边")
        return g, node_names

    def _leiden_partition(self, g: ig.Graph) -> List[List[int]]:
        """
        使用 Leiden 算法进行社区划分

        Args:
            g: igraph 图对象

        Returns:
            社区列表，每个社区包含节点索引列表
        """
        try:
            # 尝试不同的参数组合
            partition = None

            # 方式1: 使用 resolution_parameter (新版本)
            try:
                partition = find_partition(
                    g,
                    ModularityVertexPartition,
                    weights='weight',
                    resolution_parameter=self.resolution
                )
                logger.info(f"Leiden 划分成功 (resolution_parameter={self.resolution})")
            except TypeError:
                # 方式2: 使用 resolution (旧版本)
                try:
                    partition = find_partition(
                        g,
                        ModularityVertexPartition,
                        weights='weight',
                        resolution=self.resolution
                    )
                    logger.info(f"Leiden 划分成功 (resolution={self.resolution})")
                except TypeError:
                    # 方式3: 不使用 resolution 参数
                    partition = find_partition(
                        g,
                        ModularityVertexPartition,
                        weights='weight'
                    )
                    logger.info("Leiden 划分成功 (默认参数)")

            # 转换为节点索引列表
            communities = [list(community) for community in partition]
            logger.info(f"Leiden 划分完成: {len(communities)} 个社区")

            # 计算模块度
            modularity = g.modularity(partition, weights='weight')
            logger.info(f"模块度: {modularity:.4f}")

            return communities

        except Exception as e:
            logger.error(f"Leiden 算法失败: {e}")
            raise RuntimeError(f"Leiden 社区检测失败: {e}")

    def _build_communities(
            self,
            g: ig.Graph,
            communities: List[List[int]],
            node_names: List[str]
    ) -> Dict[int, Community]:
        """构建社区对象"""
        result = {}

        for comm_id, node_indices in enumerate(communities):
            # 获取实体名称列表
            entities_list = [node_names[idx] for idx in node_indices]

            # 计算社区密度
            subgraph = g.subgraph(node_indices)
            density = subgraph.density() if subgraph.vcount() > 1 else 1.0

            # 计算模块度贡献
            modularity = self._calculate_modularity_contrib(g, node_indices)

            result[comm_id] = Community(
                id=comm_id,
                entities=entities_list,
                size=len(entities_list),
                density=density,
                modularity=modularity,
                level=0
            )

        return result

    def _calculate_modularity_contrib(self, g: ig.Graph, community_nodes: List[int]) -> float:
        """计算社区内的模块度贡献"""
        if g.ecount() == 0:
            return 0.0

        community_set = set(community_nodes)

        # 计算社区内部边权重和
        internal_weight = 0.0
        total_weight = 0.0

        for edge in g.es:
            weight = edge['weight'] if 'weight' in edge.attributes() else 1.0
            total_weight += weight

            if edge.source in community_set and edge.target in community_set:
                internal_weight += weight

        if total_weight == 0:
            return 0.0

        # 计算期望内部边权重
        degree_sum = 0.0

        # 检查是否有权重属性
        has_weights = 'weight' in g.es.attributes()

        for v in community_nodes:
            try:
                # 尝试不同的参数名
                if has_weights:
                    # 先尝试 weight（单数）
                    try:
                        degree_sum += g.degree(v, weight='weight')
                    except TypeError:
                        try:
                            # 尝试 weights（复数）
                            degree_sum += g.degree(v, weights='weight')
                        except TypeError:
                            # 都不行，使用无权重版本
                            degree_sum += g.degree(v)
                else:
                    degree_sum += g.degree(v)
            except Exception as e:
                # 任何错误都回退到无权重版本
                degree_sum += g.degree(v)

        expected = (degree_sum ** 2) / (2 * total_weight)

        if expected == 0:
            return internal_weight / total_weight

        return max(0.0, min(1.0, (internal_weight / total_weight) - (expected / total_weight)))

    def _compute_community_summaries(self):
        """计算社区摘要和关键词"""
        for comm in self.communities.values():
            keywords = self._extract_keywords(comm.entities)
            comm.keywords = keywords[:10]

            # 生成摘要
            if comm.modularity > 0.3:
                quality = "高内聚"
            elif comm.modularity > 0.1:
                quality = "中等内聚"
            else:
                quality = "松散"

            comm.summary = (
                f"社区包含 {comm.size} 个相关实体，{quality}（模块度={comm.modularity:.3f}），"
                f"密度={comm.density:.2f}，主要涉及 {', '.join(keywords[:3]) if keywords else '未识别主题'}"
            )

    def _extract_keywords(self, entities: List[str]) -> List[str]:
        """从实体名称中提取关键词"""
        words = []
        for entity in entities:
            # 提取2-3个字符的词
            for length in [2, 3]:
                for i in range(len(entity) - length + 1):
                    word = entity[i:i + length]
                    if len(word) >= 2 and word not in ['公司', '有限', '股份', '有限公', '限公司']:
                        words.append(word)

        # 统计词频
        from collections import Counter
        word_counts = Counter(words)

        # 返回高频词
        return [word for word, _ in word_counts.most_common(15)]

    def get_hierarchy(self, max_levels: int = 3) -> List[Dict[int, Community]]:
        """
        获取社区层次结构

        Args:
            max_levels: 最大层次数

        Returns:
            分层社区列表
        """
        if not self.communities:
            return []

        hierarchy = []
        hierarchy.append(self.communities.copy())

        if len(self.communities) <= 1:
            return hierarchy

        current_communities = self.communities

        for level in range(1, max_levels):
            if len(current_communities) <= 2:
                break

            aggregated = self._aggregate_communities(current_communities, level)
            hierarchy.append(aggregated)
            current_communities = aggregated

        return hierarchy

    def _aggregate_communities(
            self,
            communities: Dict[int, Community],
            level: int
    ) -> Dict[int, Community]:
        """聚合社区"""
        comm_list = list(communities.values())
        similarities = []

        for i, c1 in enumerate(comm_list):
            for j, c2 in enumerate(comm_list[i + 1:], i + 1):
                keywords1 = set(c1.keywords)
                keywords2 = set(c2.keywords)
                if keywords1 and keywords2:
                    overlap = len(keywords1 & keywords2)
                    similarity = overlap / max(len(keywords1), len(keywords2))
                    if similarity > 0.1:
                        similarities.append((c1.id, c2.id, similarity))

        similarities.sort(key=lambda x: x[2], reverse=True)
        merged = set()
        aggregated = {}

        for s1, s2, sim in similarities:
            if s1 in merged or s2 in merged:
                continue

            c1 = communities[s1]
            c2 = communities[s2]

            new_id = f"agg_{level}_{s1}_{s2}"
            merged_entities = list(set(c1.entities + c2.entities))
            merged_keywords = list(set(c1.keywords + c2.keywords))

            aggregated[new_id] = Community(
                id=new_id,
                entities=merged_entities,
                size=len(merged_entities),
                density=(c1.density + c2.density) / 2,
                modularity=(c1.modularity + c2.modularity) / 2,
                keywords=merged_keywords,
                level=level
            )
            merged.add(s1)
            merged.add(s2)

        for c in communities.values():
            if c.id not in merged:
                aggregated[c.id] = c
                aggregated[c.id].level = level

        return aggregated

    def get_community_info(self, community_id: int) -> Dict:
        """获取社区详细信息"""
        if community_id not in self.communities:
            return {}

        comm = self.communities[community_id]
        return {
            "id": comm.id,
            "size": comm.size,
            "density": comm.density,
            "modularity": comm.modularity,
            "entities": comm.entities[:20],
            "total_entities": comm.size,
            "keywords": comm.keywords,
            "summary": comm.summary,
            "level": comm.level
        }

    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取当前使用的算法信息"""
        return {
            "algorithm": "leiden",
            "leiden_available": LEIDEN_AVAILABLE,
            "min_community_size": self.min_community_size,
            "resolution": self.resolution
        }


__all__ = ['CommunityDetector', 'Community']