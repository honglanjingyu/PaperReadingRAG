# app/service/core/graphrag/community.py
"""社区发现模块 - Leiden 算法"""

import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 导入 Leiden 算法
try:
    import igraph as ig
    from leidenalg import find_partition, ModularityVertexPartition
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logger.warning("Leiden 算法不可用，请安装: pip install igraph leidenalg")


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
    """社区发现器 - Leiden 算法"""

    def __init__(self, min_community_size: int = 2, resolution: float = 1.0):
        self.min_community_size = min_community_size
        self.resolution = resolution
        self.communities: Dict[int, Community] = {}

        if not LEIDEN_AVAILABLE:
            logger.warning("Leiden 不可用，将使用连通分量算法")
        else:
            logger.info(f"CommunityDetector 初始化: Leiden, resolution={resolution}")

    def detect_communities(self, entities: List[Dict], relations: List[Dict]) -> Dict[int, Community]:
        """检测实体社区"""
        if not entities or not relations:
            return {}

        if LEIDEN_AVAILABLE:
            return self._detect_with_leiden(entities, relations)
        else:
            return self._detect_with_components(entities, relations)

    def _detect_with_leiden(self, entities: List[Dict], relations: List[Dict]) -> Dict[int, Community]:
        """使用 Leiden 算法检测社区"""
        # 构建图
        node_names = [e["name"] for e in entities]
        node_index = {name: i for i, name in enumerate(node_names)}

        # 构建边
        edges = []
        weights = []
        for r in relations:
            src = r.get("source")
            tgt = r.get("target")
            if src in node_index and tgt in node_index:
                edges.append((node_index[src], node_index[tgt]))
                weights.append(r.get("weight", 1.0))

        if len(edges) == 0:
            logger.warning("没有有效的边")
            return {}

        # 创建 igraph
        g = ig.Graph()
        g.add_vertices(len(node_names))
        g.add_edges(edges)
        g.es['weight'] = weights

        # Leiden 分区
        try:
            partition = find_partition(g, ModularityVertexPartition, weights='weight', resolution_parameter=self.resolution)
        except TypeError:
            try:
                partition = find_partition(g, ModularityVertexPartition, weights='weight', resolution=self.resolution)
            except TypeError:
                partition = find_partition(g, ModularityVertexPartition, weights='weight')

        # 构建社区
        self.communities = {}
        for i, community_nodes in enumerate(partition):
            if len(community_nodes) < self.min_community_size:
                continue

            entities_list = [node_names[idx] for idx in community_nodes]

            # 计算密度
            subgraph = g.subgraph(community_nodes)
            density = subgraph.density() if subgraph.vcount() > 1 else 1.0

            self.communities[i] = Community(
                id=i,
                entities=entities_list,
                size=len(entities_list),
                density=density,
                modularity=partition.quality if hasattr(partition, 'quality') else 0
            )
            self._add_keywords(self.communities[i])

        logger.info(f"Leiden 社区检测完成: {len(self.communities)} 个社区")
        return self.communities

    def _detect_with_components(self, entities: List[Dict], relations: List[Dict]) -> Dict[int, Community]:
        """降级方案：连通分量"""
        graph = defaultdict(set)
        for r in relations:
            s, t = r.get("source", ""), r.get("target", "")
            if s and t:
                graph[s].add(t)
                graph[t].add(s)

        visited = set()
        communities = []

        for e in entities:
            name = e.get("name")
            if not name or name in visited:
                continue

            queue, comp = [name], []
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                queue.extend([n for n in graph.get(node, set()) if n not in visited])

            if len(comp) >= self.min_community_size:
                communities.append(comp)

        self.communities = {}
        for i, comp in enumerate(communities):
            self.communities[i] = Community(id=i, entities=comp, size=len(comp))
            self._add_keywords(self.communities[i])

        return self.communities

    def _add_keywords(self, comm: Community):
        """添加关键词和摘要"""
        words = []
        for e in comm.entities[:10]:
            for w in e[:3] if len(e) >= 2 else []:
                if len(w) >= 2:
                    words.append(w)

        from collections import Counter
        comm.keywords = [w for w, _ in Counter(words).most_common(5)]
        comm.summary = f"社区包含 {comm.size} 个实体，密度={comm.density:.2f}，涉及 {', '.join(comm.keywords[:3]) if comm.keywords else '未识别主题'}"

    def get_community_info(self, community_id: int) -> Dict:
        comm = self.communities.get(community_id)
        if not comm:
            return {}
        return {
            "id": comm.id, "size": comm.size, "density": comm.density,
            "entities": comm.entities[:20], "keywords": comm.keywords,
            "summary": comm.summary, "modularity": comm.modularity
        }

    def get_algorithm_info(self) -> Dict:
        return {"algorithm": "leiden" if LEIDEN_AVAILABLE else "components", "leiden_available": LEIDEN_AVAILABLE}


__all__ = ['CommunityDetector', 'Community']