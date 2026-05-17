# app/service/core/graphrag/core.py
"""实体关系提取器 - 使用 YAML 配置"""

import re
import logging
import json
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

# 导入配置管理器
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
        return {"name": self.name, "type": self.type, "frequency": self.frequency}


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
    """实体关系提取器 - 使用 YAML 配置的规则 + LLM"""

    def __init__(self, use_llm: bool = True, llm_service=None,
                 min_frequency: int = 1, max_entities: int = 100):
        self.use_llm = use_llm
        self.llm_service = llm_service
        self.min_frequency = min_frequency
        self.max_entities = max_entities

        # 获取配置管理器
        self.config = get_graphrag_config()

        # 先初始化为空
        self.entity_patterns = {}
        self.relation_patterns = []

        # 从配置加载实体模式
        self.entity_patterns = self._load_entity_patterns()

        # 从配置加载关系模式
        self.relation_patterns = self._load_relation_patterns()

        # 实体类型优先级
        self.type_priority = self.config.get_type_priority_order()

        # 获取别名映射
        self.aliases = self.config.get_entity_aliases()
        self.case_insensitive = self.config.is_case_insensitive()

        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        logger.info(f"EntityExtractor 初始化: use_llm={use_llm}, max_entities={max_entities}, "
                    f"实体类型数={len(self.entity_patterns)}, 关系模式数={len(self.relation_patterns)}")

    def _load_entity_patterns(self) -> Dict[str, List[re.Pattern]]:
        """从配置加载实体模式"""
        compiled = self.config.get_compiled_entity_patterns()

        if compiled and isinstance(compiled, dict):
            logger.info(f"从配置加载实体模式: {len(compiled)} 种类型")
            return compiled

        # 降级：使用默认模式
        logger.warning("配置中无实体模式，使用默认模式")
        return {
            "ORGANIZATION": [
                re.compile(r'([\u4e00-\u9fff]{2,}(?:公司|集团|有限|股份|银行|基金|证券|保险|投资|资本|科技|技术))'),
                re.compile(r'([A-Z][a-z]+(?:Inc|Corp|Ltd|Company|Group))'),
            ],
            "PERSON": [
                re.compile(r'([\u4e00-\u9fff]{2,4})(?:先生|女士|博士|教授|经理|CEO|总裁|董事长)'),
                re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)'),
            ],
            "PRODUCT": [
                re.compile(r'([\u4e00-\u9fff]{2,}(?:电池|芯片|系统|平台|软件|硬件|产品|解决方案))'),
            ],
            "TECHNOLOGY": [
                re.compile(r'([\u4e00-\u9fff]{2,}(?:技术|算法|模型|框架|架构|方案|引擎))'),
                re.compile(r'([A-Z]{2,}(?:[\s-][A-Z0-9]+)*)'),
            ],
        }

    def _load_relation_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """从配置加载关系模式"""
        compiled = self.config.get_compiled_relation_patterns()

        if compiled is None:
            logger.warning("配置中无关系模式，使用默认模式")
            return self._get_default_relation_patterns()

        # 如果是字典格式，转换为列表
        if isinstance(compiled, dict):
            result = []
            for pattern_str, rel_type in compiled.items():
                try:
                    # 如果 pattern_str 已经是编译好的 Pattern
                    if isinstance(pattern_str, re.Pattern):
                        result.append((pattern_str, rel_type))
                    else:
                        result.append((re.compile(pattern_str), rel_type))
                except re.error as e:
                    logger.warning(f"编译关系模式失败: {pattern_str} -> {e}")
            if result:
                logger.info(f"从配置加载关系模式: {len(result)} 条")
                return result

        # 如果已经是列表格式
        if isinstance(compiled, list) and compiled:
            logger.info(f"从配置加载关系模式: {len(compiled)} 条")
            return compiled

        # 降级：使用默认模式
        return self._get_default_relation_patterns()

    def _get_default_relation_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """获取默认关系模式"""
        return [
            (re.compile(r'([\u4e00-\u9fff]{2,})[的]?(?:董事长|CEO|总经理)(?:是|为)([\u4e00-\u9fff]{2,})'), "LEADER_OF"),
            (re.compile(r'([\u4e00-\u9fff]{2,})(?:与|和)([\u4e00-\u9fff]{2,})(?:合作|竞争)'), "RELATED_TO"),
        ]

    def extract_from_text(self, text: str, doc_name: str = "") -> Tuple[List[Entity], List[Relation]]:
        """从文本提取实体和关系"""
        if not text:
            return [], []

        # 1. 规则提取（使用配置的模式）
        rule_entities = self._extract_entities_rules(text)
        rule_relations = self._extract_relations_rules(text)

        # 2. LLM 增强
        if self.use_llm and self.llm_service:
            llm_entities, llm_relations = self._extract_with_llm(text)
            rule_entities = self._merge_entities(rule_entities, llm_entities)
            rule_relations.extend(llm_relations)

        # 3. 去重合并
        merged_entities = self._deduplicate_entities(rule_entities)
        merged_relations = self._deduplicate_relations(rule_relations)

        self.entities = {e.name: e for e in merged_entities}
        self.relations = merged_relations

        logger.info(f"实体提取完成: {len(merged_entities)} 实体, {len(merged_relations)} 关系")
        return merged_entities, merged_relations

    def _extract_entities_rules(self, text: str) -> List[Entity]:
        """使用配置的规则提取实体"""
        entities = []
        text_sample = text[:8000]  # 限制长度

        # 确保 entity_patterns 是字典
        if not isinstance(self.entity_patterns, dict):
            logger.warning(f"entity_patterns 格式错误: {type(self.entity_patterns)}")
            return entities

        # 按优先级顺序提取
        for entity_type in self.type_priority:
            patterns = self.entity_patterns.get(entity_type, [])
            # 确保 patterns 是列表
            if not isinstance(patterns, list):
                patterns = []
            for pattern in patterns:
                try:
                    for match in pattern.finditer(text_sample):
                        # 获取匹配的实体名称（通常是第一个捕获组）
                        name = match.group(1) if match.groups() else match.group(0)
                        name = name.strip()

                        # 应用别名标准化
                        name = self.config.normalize_entity_name(name)

                        # 长度过滤
                        min_len = self.config.get_min_entity_length()
                        max_len = self.config.get_max_entity_length()

                        if min_len <= len(name) <= max_len:
                            entities.append(Entity(name=name, type=entity_type))
                except Exception as e:
                    logger.warning(f"实体匹配失败 {entity_type}: {e}")
                    continue

        # 限制实体数量
        max_entities = self.config.get_max_entities_per_doc()
        if len(entities) > max_entities:
            entities = entities[:max_entities]

        return entities

    def _extract_relations_rules(self, text: str) -> List[Relation]:
        """使用配置的规则提取关系"""
        relations = []
        text_sample = text[:8000]

        # 确保 relation_patterns 是列表
        if not isinstance(self.relation_patterns, list):
            logger.warning(f"relation_patterns 格式错误，类型: {type(self.relation_patterns)}")
            return relations

        for pattern, rel_type in self.relation_patterns:
            try:
                for match in pattern.finditer(text_sample):
                    groups = match.groups()
                    if len(groups) >= 2:
                        source = groups[0].strip()
                        target = groups[-1].strip()

                        # 标准化实体名称
                        source = self.config.normalize_entity_name(source)
                        target = self.config.normalize_entity_name(target)

                        if source and target and source != target:
                            # 获取关系权重
                            weight = self.config.get_relation_weight(rel_type)
                            relations.append(Relation(
                                source=source,
                                target=target,
                                relation_type=rel_type,
                                weight=weight
                            ))
            except Exception as e:
                logger.warning(f"关系匹配失败 {rel_type}: {e}")
                continue

        return relations

    def _extract_with_llm(self, text: str, max_retries: int = 2) -> Tuple[List[Entity], List[Relation]]:
        """LLM 提取实体和关系"""
        if not self.llm_service:
            return [], []

        text_preview = text[:4000] + "..." if len(text) > 4000 else text

        # 获取实体类型列表用于提示
        entity_types = list(self.entity_patterns.keys()) if self.entity_patterns else ["COMPANY", "PERSON", "PRODUCT"]
        relation_types = [rt for _, rt in self.relation_patterns[:10]] if self.relation_patterns else ["RELATED_TO"]

        prompt = f"""请分析以下文档，提取关键实体和关系。

## 文档内容
{text_preview}

## 实体类型
{', '.join(entity_types[:8])}

## 关系类型
{', '.join(relation_types[:8])}

## 输出格式（只输出 JSON）
{{
    "entities": [
        {{"name": "实体名", "type": "实体类型"}}
    ],
    "relations": [
        {{"source": "源实体", "target": "目标实体", "relation_type": "关系类型"}}
    ]
}}

JSON:"""

        for attempt in range(max_retries + 1):
            try:
                response = self.llm_service.generate([{"role": "user", "content": prompt}])
                if not response:
                    continue

                # 提取 JSON
                json_str = self._extract_json(response)
                data = json.loads(json_str)

                entities = []
                for e in data.get("entities", []):
                    name = e.get("name", "").strip()
                    if name:
                        name = self.config.normalize_entity_name(name)
                        entities.append(Entity(
                            name=name,
                            type=e.get("type", "CONCEPT")
                        ))

                relations = []
                for r in data.get("relations", []):
                    source = r.get("source", "").strip()
                    target = r.get("target", "").strip()
                    if source and target and source != target:
                        source = self.config.normalize_entity_name(source)
                        target = self.config.normalize_entity_name(target)
                        relations.append(Relation(
                            source=source,
                            target=target,
                            relation_type=r.get("relation_type", "RELATED_TO")
                        ))

                logger.info(f"LLM 提取: {len(entities)} 实体, {len(relations)} 关系")
                return entities, relations

            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败 (尝试 {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"LLM 提取失败: {e}")

        return [], []

    def _extract_json(self, response: str) -> str:
        """从响应中提取 JSON"""
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1:
            return response[start:end + 1]
        return "{}"

    def _merge_entities(self, e1: List[Entity], e2: List[Entity]) -> List[Entity]:
        """合并实体列表"""
        merged = {e.name: e for e in e1}
        for e in e2:
            if e.name in merged:
                merged[e.name].frequency += e.frequency
            else:
                merged[e.name] = e
        return list(merged.values())

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """实体去重"""
        unique = {}
        for e in entities:
            name = e.name.strip()
            if name in unique:
                unique[name].frequency += e.frequency
            else:
                e.name = name
                unique[name] = e

        result = list(unique.values())
        if self.min_frequency > 1:
            result = [e for e in result if e.frequency >= self.min_frequency]
        if self.max_entities and len(result) > self.max_entities:
            result.sort(key=lambda x: x.frequency, reverse=True)
            result = result[:self.max_entities]
        return result

    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """关系去重"""
        unique = {}
        for r in relations:
            key = f"{r.source}|{r.target}|{r.relation_type}"
            if key in unique:
                unique[key].weight += r.weight
            else:
                unique[key] = r
        return list(unique.values())

    def to_dict(self) -> Dict:
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations],
            "entity_count": len(self.entities),
            "relation_count": len(self.relations)
        }


__all__ = ['EntityExtractor', 'Entity', 'Relation']