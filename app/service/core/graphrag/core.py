# app/service/core/graphrag/core.py
"""实体关系提取器 - 支持 LLM 增强"""

import re
import logging
import json
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

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
    """实体关系提取器 - 规则 + LLM"""

    def __init__(self, use_llm: bool = True, llm_service=None, min_frequency: int = 1, max_entities: int = 100):
        self.use_llm = use_llm
        self.llm_service = llm_service
        self.min_frequency = min_frequency
        self.max_entities = max_entities

        # 规则模式
        self.entity_patterns = {
            "PERSON": [re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)')],
            "ORGANIZATION": [re.compile(r'([\u4e00-\u9fff]{2,}(?:公司|集团|有限|股份|银行|大学|研究院|中心))')],
            "LOCATION": [re.compile(r'([\u4e00-\u9fff]{2,}(?:市|省|区|县|国|州|城))')],
            "CONCEPT": [re.compile(r'([\u4e00-\u9fff]{2,}(?:技术|系统|模型|方法|算法|框架|平台|数据|服务))')],
            "PRODUCT": [re.compile(r'([\u4e00-\u9fff]{2,}(?:产品|服务|工具|软件|平台))')],
        }
        self.relation_patterns = [
            (re.compile(r'([\u4e00-\u9fff]{2,})[的]?([公司|团队|部门])[包含|有]?([\u4e00-\u9fff]{2,})'), "HAS"),
            (re.compile(r'([\u4e00-\u9fff]{2,})[与和及]([\u4e00-\u9fff]{2,})'), "RELATED_TO"),
        ]

        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        logger.info(f"EntityExtractor 初始化: use_llm={use_llm}, max_entities={max_entities}")

    def extract_from_text(self, text: str, doc_name: str = "") -> Tuple[List[Entity], List[Relation]]:
        """从文本提取实体和关系"""
        if not text:
            return [], []

        # 1. 规则提取
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
        """规则提取实体"""
        entities = []
        text_sample = text[:8000]  # 限制长度
        for etype, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text_sample):
                    name = match.group(1).strip()
                    if 2 <= len(name) <= 50:
                        entities.append(Entity(name=name, type=etype))
        return entities

    def _extract_relations_rules(self, text: str) -> List[Relation]:
        """规则提取关系"""
        relations = []
        for pattern, rel_type in self.relation_patterns:
            for match in pattern.finditer(text[:8000]):
                groups = match.groups()
                if len(groups) >= 2:
                    source = groups[0].strip()
                    target = groups[-1].strip()
                    if source and target:
                        relations.append(Relation(source=source, target=target, relation_type=rel_type))
        return relations

    def _extract_with_llm(self, text: str, max_retries: int = 2) -> Tuple[List[Entity], List[Relation]]:
        """LLM 提取实体和关系"""
        if not self.llm_service:
            return [], []

        text_preview = text[:4000] + "..." if len(text) > 4000 else text

        prompt = f"""请分析以下文档，提取关键实体和关系。

## 文档内容
{text_preview}

## 输出格式（只输出 JSON）
{{
    "entities": [
        {{"name": "实体名", "type": "PERSON/ORGANIZATION/CONCEPT"}}
    ],
    "relations": [
        {{"source": "源实体", "target": "目标实体", "relation_type": "RELATED_TO"}}
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

                entities = [Entity(name=e["name"].strip(), type=e.get("type", "CONCEPT"))
                            for e in data.get("entities", []) if e.get("name")]
                relations = [Relation(source=r["source"].strip(), target=r["target"].strip(),
                                      relation_type=r.get("relation_type", "RELATED_TO"))
                             for r in data.get("relations", []) if r.get("source") and r.get("target")]

                logger.info(f"LLM 提取: {len(entities)} 实体, {len(relations)} 关系")
                return entities, relations

            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败 (尝试 {attempt+1}): {e}")
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