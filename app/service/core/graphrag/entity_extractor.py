# app/service/core/graphrag/entity_extractor.py - 修改后

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

# 导入配置加载器
from app.service.graphrag_configs import get_graphrag_config

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体数据结构"""
    name: str
    type: str
    mentions: List[str] = field(default_factory=list)
    frequency: int = 1

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type,
            "mentions": self.mentions,
            "frequency": self.frequency
        }


@dataclass
class Relation:
    """关系数据结构"""
    source: str
    target: str
    relation_type: str
    weight: float = 1.0
    evidence: str = ""

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "weight": self.weight,
            "evidence": self.evidence[:200] if self.evidence else ""
        }


class EntityExtractor:
    """实体关系提取器 - 使用外置配置"""

    def __init__(self, use_llm: bool = True, llm_service=None, min_frequency: int = 1, max_entities: int = 100):
        self.use_llm = use_llm
        self.llm_service = llm_service
        self.min_frequency = min_frequency
        self.max_entities = max_entities

        # 加载配置
        self.config = get_graphrag_config()

        # 从配置加载实体模式
        self.ENTITY_PATTERNS = self.config.get_entity_patterns()
        self.compiled_entity_patterns = self.config.get_compiled_entity_patterns()

        # 从配置加载关系模式
        self.compiled_relations = self.config.get_compiled_relation_patterns()
        self.RELATION_PATTERNS = self.config.get_relation_patterns()

        # 获取实体长度限制
        self.min_entity_length, self.max_entity_length = self.config.get_entity_length_limits()

        # 获取别名配置
        self.entity_aliases = self.config.get_entity_aliases()

        # 实体和关系存储
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_graph: Dict[str, Set[str]] = defaultdict(set)

        logger.info(f"EntityExtractor 初始化完成: use_llm={use_llm}, "
                    f"min_frequency={min_frequency}, max_entities={max_entities}, "
                    f"实体类型数={len(self.ENTITY_PATTERNS)}, "
                    f"关系模式数={len(self.RELATION_PATTERNS)}")

    def normalize_entity_name(self, name: str) -> str:
        """将实体名称标准化（使用配置中的别名映射）"""
        return self.config.normalize_entity_name(name)

    def extract_from_text(self, text: str, doc_name: str = "") -> Tuple[List[Entity], List[Relation]]:
        """从文本中提取实体和关系"""
        if not text:
            return [], []

        # 1. 使用规则提取实体
        rule_entities = self._extract_entities_rules(text)

        # 2. 使用规则提取关系
        rule_relations = self._extract_relations_rules(text)

        # 3. 如果启用 LLM，进行增强
        if self.use_llm and self.llm_service:
            llm_entities, llm_relations = self._extract_with_llm(text)
            rule_entities = self._merge_entities(rule_entities, llm_entities)
            rule_relations.extend(llm_relations)

        # 4. 去重和合并
        merged_entities = self._deduplicate_entities(rule_entities)
        merged_relations = self._deduplicate_relations(rule_relations)

        # 5. 标准化实体名称
        for e in merged_entities:
            e.name = self.normalize_entity_name(e.name)

        logger.info(f"实体提取完成: {len(merged_entities)} 个实体, {len(merged_relations)} 个关系")

        # 存储到实例变量
        self.entities = {e.name: e for e in merged_entities}
        self.relations = merged_relations

        # 构建关系图
        self._build_entity_graph()

        return merged_entities, merged_relations

    def _extract_entities_rules(self, text: str) -> List[Entity]:
        """使用规则提取实体"""
        entities = []

        for entity_type, patterns in self.compiled_entity_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    entity_name = match.group(1).strip()
                    # 应用长度限制
                    if entity_name and self.min_entity_length <= len(entity_name) <= self.max_entity_length:
                        entities.append(Entity(
                            name=entity_name,
                            type=entity_type,
                            mentions=[entity_name]
                        ))

        return entities

    def _extract_relations_rules(self, text: str) -> List[Relation]:
        """使用规则提取关系"""
        relations = []

        for pattern, rel_type in self.compiled_relations:
            matches = pattern.finditer(text)
            for match in matches:
                source = match.group(1).strip() if match.lastindex >= 2 else ""
                target = match.group(2).strip() if match.lastindex >= 2 else ""

                if source and target:
                    relations.append(Relation(
                        source=source,
                        target=target,
                        relation_type=rel_type,
                        evidence=match.group(0)
                    ))

        return relations

    def _extract_with_llm(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """使用 LLM 提取实体和关系"""
        if not self.llm_service:
            return [], []

        # 限制文本长度
        max_len = 4000
        text_preview = text[:max_len] + "..." if len(text) > max_len else text

        prompt = f"""请分析以下文档内容，提取其中的关键实体和关系。

## 实体类型
- PERSON: 人物、作者、研究者
- ORGANIZATION: 公司、机构、组织、政府部门
- LOCATION: 地理位置、城市、国家
- CONCEPT: 核心概念、技术名词、专业术语
- PRODUCT: 产品、服务、工具
- DATE: 时间、年份、时期
- NUMBER: 重要数字、指标

## 关系类型
- WORKS_FOR: 某人工作在某个组织
- LOCATED_IN: 位于某地
- PRODUCES: 生产/提供某产品
- RELATED_TO: 相关关系
- PART_OF: 组成部分
- LEADS_TO: 导致/促使

## 文档内容
{text_preview}

## 输出格式
请以 JSON 格式输出：
{{
    "entities": [
        {{"name": "实体名", "type": "实体类型", "description": "简短描述"}}
    ],
    "relations": [
        {{"source": "源实体", "target": "目标实体", "relation_type": "关系类型", "evidence": "原文证据"}}
    ]
}}

只输出 JSON，不要有其他内容："""

        try:
            response = self.llm_service.generate([{"role": "user", "content": prompt}])

            if response:
                import json
                # 提取 JSON
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())

                    entities = [
                        Entity(name=e["name"], type=e.get("type", "CONCEPT"), mentions=[])
                        for e in data.get("entities", [])
                    ]

                    relations = [
                        Relation(
                            source=r["source"],
                            target=r["target"],
                            relation_type=r.get("relation_type", "RELATED_TO"),
                            evidence=r.get("evidence", "")
                        )
                        for r in data.get("relations", [])
                    ]

                    logger.info(f"LLM 提取: {len(entities)} 个实体, {len(relations)} 个关系")
                    return entities, relations

        except Exception as e:
            logger.error(f"LLM 提取失败: {e}")

        return [], []

    def _merge_entities(self, entities1: List[Entity], entities2: List[Entity]) -> List[Entity]:
        """合并两组实体"""
        merged = {e.name: e for e in entities1}

        for e in entities2:
            if e.name in merged:
                merged[e.name].frequency += e.frequency
                merged[e.name].mentions.extend(e.mentions)
            else:
                merged[e.name] = e

        return list(merged.values())

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        实体去重（合并相似名称）并应用频率过滤和数量限制
        """
        # 简单去重：基于名称
        unique = {}

        for e in entities:
            # 规范化名称（去除空格）
            normalized = e.name.strip()

            if normalized in unique:
                unique[normalized].frequency += e.frequency
                if e.mentions:
                    unique[normalized].mentions.extend(e.mentions)
            else:
                e.name = normalized
                unique[normalized] = e

        # 过滤低频实体（使用 min_frequency）
        if self.min_frequency > 1:
            filtered = [e for e in unique.values() if e.frequency >= self.min_frequency]
        else:
            filtered = list(unique.values())

        # 限制最大实体数量（使用 max_entities）
        if self.max_entities and len(filtered) > self.max_entities:
            # 按频率排序，保留高频实体
            filtered.sort(key=lambda x: x.frequency, reverse=True)
            filtered = filtered[:self.max_entities]

        logger.debug(
            f"实体去重: {len(entities)} -> {len(unique)} -> {len(filtered)} (min_freq={self.min_frequency}, max={self.max_entities})")
        return filtered

    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """关系去重"""
        unique = {}

        for r in relations:
            key = f"{r.source}|{r.target}|{r.relation_type}"

            if key in unique:
                unique[key].weight += r.weight
                if not unique[key].evidence and r.evidence:
                    unique[key].evidence = r.evidence
            else:
                unique[key] = r
                unique[key].weight = 1.0

        return list(unique.values())

    def _build_entity_graph(self):
        """构建实体关系图"""
        self.entity_graph.clear()

        for r in self.relations:
            if r.source in self.entities:
                self.entity_graph[r.source].add(r.target)
            if r.target in self.entities:
                self.entity_graph[r.target].add(r.source)

    def get_entity_neighbors(self, entity_name: str, max_depth: int = 2) -> List[Tuple[str, int]]:
        """获取实体的邻居（广度优先）"""
        visited = set()
        queue = [(entity_name, 0)]
        neighbors = []

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            for neighbor in self.entity_graph.get(current, set()):
                if neighbor not in visited:
                    neighbors.append((neighbor, depth + 1))
                    queue.append((neighbor, depth + 1))

        return neighbors

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations],
            "entity_count": len(self.entities),
            "relation_count": len(self.relations)
        }