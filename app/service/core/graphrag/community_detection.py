# app/service/core/graphrag/community_detection.py
"""
社区发现模块 - 使用 Louvain 算法识别实体社区
"""

import logging
import networkx as nx
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import community as community_louvain

    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain 未安装，将使用简化社区检测")


@dataclass
class Community:
    """社区数据结构"""
    id: int
    entities: List[str]
    size: int
    density: float = 0.0
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    level: int = 0  # 层次级别


class CommunityDetector:
    """
    社区发现器
    使用 Louvain 算法识别实体网络中的社区结构
    """
    def __init__(self, use_louvain: bool = True, min_community_size: int = 2):
        """
        初始化社区检测器

        Args:
            use_louvain: 是否使用 Louvain 算法
            min_community_size: 最小社区大小（少于该值的社区将被过滤）
        """
        self.use_louvain = use_louvain and LOUVAIN_AVAILABLE
        self.min_community_size = min_community_size  # 新增
        self.communities: Dict[int, Community] = {}

        logger.info(
            f"CommunityDetector 初始化, use_louvain={self.use_louvain}, min_community_size={min_community_size}")

    def detect_communities(
            self,
            entities: List[Dict],
            relations: List[Dict]
    ) -> Dict[int, Community]:
        """
        检测实体社区
        """
        if not entities or not relations:
            return {}

        # 构建图
        G = self._build_graph(entities, relations)

        if G.number_of_nodes() < 2:
            return {}

        # 社区检测
        if self.use_louvain:
            partition = self._louvain_partition(G)
        else:
            partition = self._simple_partition(G)

        # 构建社区对象
        self.communities = self._build_communities(G, partition)

        # 过滤小社区（新增）
        if self.min_community_size > 1:
            self.communities = {
                cid: c for cid, c in self.communities.items()
                if c.size >= self.min_community_size
            }

        # 计算社区摘要
        self._compute_community_summaries()

        logger.info(f"社区检测完成: {len(self.communities)} 个社区")
        return self.communities

    def _build_graph(self, entities: List[Dict], relations: List[Dict]) -> nx.Graph:
        """构建实体关系图"""
        G = nx.Graph()

        # 添加节点（实体）
        for entity in entities:
            G.add_node(
                entity["name"],
                type=entity.get("type", "UNKNOWN"),
                frequency=entity.get("frequency", 1)
            )

        # 添加边（关系）
        for relation in relations:
            source = relation["source"]
            target = relation["target"]
            weight = relation.get("weight", 1.0)

            if G.has_node(source) and G.has_node(target):
                if G.has_edge(source, target):
                    G[source][target]["weight"] += weight
                else:
                    G.add_edge(source, target, weight=weight)

        logger.info(f"图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        return G

    def _louvain_partition(self, G: nx.Graph) -> Dict:
        """使用 Louvain 算法进行社区划分"""
        try:
            # 使用权重
            partition = community_louvain.best_partition(G, weight='weight')
            logger.info(f"Louvain 划分完成: {len(set(partition.values()))} 个社区")
            return partition
        except Exception as e:
            logger.error(f"Louvain 算法失败: {e}")
            return self._simple_partition(G)

    def _simple_partition(self, G: nx.Graph) -> Dict:
        """简化的社区划分（基于连通分量）"""
        components = list(nx.connected_components(G))
        partition = {}

        for comp_id, component in enumerate(components):
            for node in component:
                partition[node] = comp_id

        logger.info(f"连通分量划分完成: {len(components)} 个社区")
        return partition

    def _build_communities(self, G: nx.Graph, partition: Dict) -> Dict[int, Community]:
        """构建社区对象"""
        communities = {}

        # 按社区分组
        community_groups: Dict[int, List[str]] = defaultdict(list)
        for node, comm_id in partition.items():
            community_groups[comm_id].append(node)

        # 创建社区对象
        for comm_id, entities in community_groups.items():
            # 计算社区密度
            subgraph = G.subgraph(entities)
            density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 1.0

            communities[comm_id] = Community(
                id=comm_id,
                entities=entities,
                size=len(entities),
                density=density,
                level=0
            )

        return communities

    def _compute_community_summaries(self):
        """计算社区摘要和关键词"""
        for comm in self.communities.values():
            # 关键词提取（基于实体名称中的高频词）
            keywords = self._extract_keywords(comm.entities)
            comm.keywords = keywords[:10]
            comm.summary = f"包含 {comm.size} 个相关实体，主要涉及 {', '.join(keywords[:3])}"

    def _extract_keywords(self, entities: List[str]) -> List[str]:
        """从实体名称中提取关键词"""
        words = []
        for entity in entities:
            # 中文分词（简单处理）
            for i in range(len(entity) - 1):
                word = entity[i:i + 2]
                if len(word) >= 2 and word not in ['公司', '有限', '股份']:
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

        # Level 0: 原始社区
        hierarchy.append(self.communities.copy())

        # 如果只有一个社区，无法继续聚合
        if len(self.communities) <= 1:
            return hierarchy

        # Level 1+: 社区聚合
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
        """聚合社区（简单的层次聚类）"""
        # 计算社区间的相似度
        comm_list = list(communities.values())
        similarities = []

        for i, c1 in enumerate(comm_list):
            for j, c2 in enumerate(comm_list[i + 1:], i + 1):
                # 基于共同关键词的相似度
                keywords1 = set(c1.keywords)
                keywords2 = set(c2.keywords)
                if keywords1 and keywords2:
                    overlap = len(keywords1 & keywords2)
                    similarity = overlap / max(len(keywords1), len(keywords2))
                    if similarity > 0.1:
                        similarities.append((c1.id, c2.id, similarity))

        # 按相似度排序并合并
        similarities.sort(key=lambda x: x[2], reverse=True)
        merged = set()
        aggregated = {}

        for s1, s2, sim in similarities:
            if s1 in merged or s2 in merged:
                continue

            c1 = communities[s1]
            c2 = communities[s2]

            # 合并
            new_id = f"agg_{level}_{s1}_{s2}"
            merged_entities = list(set(c1.entities + c2.entities))
            merged_keywords = list(set(c1.keywords + c2.keywords))

            aggregated[new_id] = Community(
                id=new_id,
                entities=merged_entities,
                size=len(merged_entities),
                density=(c1.density + c2.density) / 2,
                keywords=merged_keywords,
                level=level
            )
            merged.add(s1)
            merged.add(s2)

        # 添加未合并的社区
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
            "entities": comm.entities[:20],
            "total_entities": comm.size,
            "keywords": comm.keywords,
            "summary": comm.summary,
            "level": comm.level
        }