# app/service/graphrag_configs/loader.py
"""统一配置加载器 - 支持所有 YAML 配置文件，包括 synonymlist"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML 未安装，请运行: pip install pyyaml")


class GraphRAGConfig:
    """GraphRAG 统一配置管理器 - 单例模式"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.config_dir = Path(__file__).parent
        self._configs: Dict[str, Dict] = {}
        self._compiled_entity_patterns: Dict[str, List[re.Pattern]] = {}
        self._compiled_relation_patterns: List[Tuple[re.Pattern, str]] = {}
        self._loaded = False

        if not YAML_AVAILABLE:
            logger.error("PyYAML 未安装，无法加载配置")
            return

        self._load_all()

    def _get_synonym_dir(self) -> Path:
        """获取同义词目录路径"""
        # 尝试多个可能的路径
        possible_paths = [
            Path(__file__).parent.parent / "synonymlist",  # app/service/synonymlist
            Path(__file__).parent.parent.parent / "synonymlist",  # app/synonymlist
            Path(__file__).parent.parent.parent.parent / "synonymlist",  # project/synonymlist
        ]

        for path in possible_paths:
            if path.exists():
                logger.debug(f"找到同义词目录: {path}")
                return path

        # 默认路径
        default_path = Path(__file__).parent.parent / "synonymlist"
        logger.warning(f"同义词目录不存在: {default_path}")
        return default_path

    def _load_all(self):
        """加载所有配置文件"""
        # GraphRAG 核心配置
        core_config_files = {
            'entity_types': self.config_dir / 'entity_types.yml',
            'relation_types': self.config_dir / 'relation_types.yml',
            'entity_patterns': self.config_dir / 'entity_patterns.yml',
            'relation_patterns': self.config_dir / 'relation_patterns.yml',
            'entity_aliases': self.config_dir / 'entity_aliases.yml',
            'query_replacements': self.config_dir / 'query_replacements.yml',
            'entity_extraction': self.config_dir / 'entity_extraction.yml',
        }

        for name, file_path in core_config_files.items():
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._configs[name] = yaml.safe_load(f) or {}
                    logger.debug(f"加载配置: {name} -> {file_path}")
                except Exception as e:
                    logger.error(f"加载配置失败 {name}: {e}")
                    self._configs[name] = {}
            else:
                logger.debug(f"配置文件不存在: {file_path}")
                self._configs[name] = {}

        # ========== 加载同义词配置（从 synonymlist 目录） ==========
        self._configs['synonyms'] = self._load_synonyms_from_dir()

        # 编译正则表达式
        self._compile_patterns()
        self._loaded = True
        logger.info(f"GraphRAG 配置加载完成，共 {len(self._configs)} 个配置组")

    def _load_synonyms_from_dir(self) -> Dict[str, List[str]]:
        """从 synonymlist 目录加载所有同义词文件"""
        synonym_dir = self._get_synonym_dir()
        all_synonyms = {}

        if not synonym_dir.exists():
            logger.warning(f"同义词目录不存在: {synonym_dir}")
            return all_synonyms

        # 加载所有 yml 文件
        yaml_files = list(synonym_dir.glob("*.yml"))

        for yaml_file in yaml_files:
            if yaml_file.name == "__init__.py":
                continue

            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)

                if not data:
                    logger.warning(f"同义词文件为空: {yaml_file}")
                    continue

                for key, values in data.items():
                    if isinstance(values, list):
                        if key in all_synonyms:
                            # 合并同义词
                            existing = set(all_synonyms[key])
                            existing.update(values)
                            all_synonyms[key] = list(existing)
                        else:
                            all_synonyms[key] = values
                    else:
                        logger.warning(f"跳过非列表格式: {yaml_file}:{key}")

                logger.debug(f"加载同义词文件: {yaml_file.name}, {len(data)} 个词条")

            except Exception as e:
                logger.error(f"加载同义词文件失败 {yaml_file}: {e}")

        logger.info(f"从 synonymlist 加载同义词完成: {len(all_synonyms)} 个词条")
        return all_synonyms

    def _compile_patterns(self):
        """编译所有正则表达式模式"""
        # 编译实体模式
        entity_patterns = self.get_entity_patterns()
        for entity_type, patterns in entity_patterns.items():
            self._compiled_entity_patterns[entity_type] = []
            if isinstance(patterns, dict) and 'patterns' in patterns:
                pattern_list = patterns['patterns']
            else:
                pattern_list = patterns if isinstance(patterns, list) else []

            for pattern_str in pattern_list:
                try:
                    # 处理 YAML 中的转义
                    pattern_str = pattern_str.replace('\\\\', '\\')
                    pattern_str = pattern_str.encode('utf-8').decode('unicode_escape')
                    self._compiled_entity_patterns[entity_type].append(re.compile(pattern_str))
                except re.error as e:
                    logger.warning(f"编译实体模式失败 {entity_type}: {pattern_str} -> {e}")

        # 编译关系模式
        for item in self._configs.get('relation_patterns', {}).get('relation_patterns', []):
            pattern_str = item.get('pattern', '')
            relation_type = item.get('relation_type', 'RELATED_TO')
            if pattern_str:
                try:
                    pattern_str = pattern_str.replace('\\\\', '\\')
                    pattern_str = pattern_str.encode('utf-8').decode('unicode_escape')
                    self._compiled_relation_patterns[pattern_str] = relation_type
                except re.error as e:
                    logger.warning(f"编译关系模式失败: {pattern_str} -> {e}")

    def reload(self):
        """重新加载所有配置"""
        self._configs.clear()
        self._compiled_entity_patterns.clear()
        self._compiled_relation_patterns.clear()
        self._load_all()

    # ========== 同义词配置（从 synonymlist 加载） ==========

    def get_all_synonyms(self) -> Dict[str, List[str]]:
        """获取所有同义词（从 synonymlist 目录加载）"""
        return self._configs.get('synonyms', {})

    def get_synonyms(self, word: str) -> List[str]:
        """获取指定词的同义词"""
        all_synonyms = self.get_all_synonyms()
        return all_synonyms.get(word, [])

    def expand_with_synonyms(self, query: str) -> str:
        """
        使用同义词扩展查询

        Args:
            query: 原始查询

        Returns:
            扩展后的查询（同义词用 OR 连接）
        """
        all_synonyms = self.get_all_synonyms()
        if not all_synonyms:
            return query

        expanded = query
        for word, synonyms in all_synonyms.items():
            if word in query:
                # 构建 OR 表达式
                synonym_str = " OR ".join(synonyms)
                expanded = expanded.replace(word, f"({word} OR {synonym_str})")

        return expanded

    # ========== 实体类型配置 ==========

    def get_entity_types(self) -> Dict[str, Dict]:
        """获取所有实体类型定义"""
        return self._configs.get('entity_types', {}).get('entity_types', {})

    def get_entity_type_info(self, entity_type: str) -> Dict:
        """获取单个实体类型信息"""
        types = self.get_entity_types()
        return types.get(entity_type, {})

    def get_entity_type_priority(self, entity_type: str) -> int:
        """获取实体类型优先级"""
        info = self.get_entity_type_info(entity_type)
        return info.get('priority', 999)

    def get_entity_type_color(self, entity_type: str) -> str:
        """获取实体类型颜色"""
        info = self.get_entity_type_info(entity_type)
        return info.get('color', '#6b7280')

    def get_entity_type_icon(self, entity_type: str) -> str:
        """获取实体类型图标"""
        info = self.get_entity_type_info(entity_type)
        return info.get('icon', 'circle')

    # ========== 关系类型配置 ==========

    def get_relation_types(self) -> Dict[str, Dict]:
        """获取所有关系类型定义"""
        return self._configs.get('relation_types', {}).get('relation_types', {})

    def get_relation_type_info(self, relation_type: str) -> Dict:
        """获取单个关系类型信息"""
        types = self.get_relation_types()
        return types.get(relation_type, {})

    def get_relation_weight(self, relation_type: str) -> float:
        """获取关系权重"""
        info = self.get_relation_type_info(relation_type)
        return info.get('weight', 1.0)

    def get_relation_direction(self, relation_type: str) -> str:
        """获取关系方向"""
        info = self.get_relation_type_info(relation_type)
        return info.get('direction', 'directed')

    # ========== 实体提取配置 ==========

    def get_entity_extraction_patterns(self) -> List[str]:
        """获取从问题中提取实体的正则表达式模式"""
        config = self._configs.get('entity_extraction', {})
        patterns = config.get('entity_extraction_patterns', [])
        # 处理 r'...' 格式
        return [p.replace("r'", "").replace("'", "") if p.startswith("r'") else p for p in patterns]

    def get_min_entity_length(self) -> int:
        """获取最小实体长度"""
        config = self._configs.get('entity_extraction', {})
        return config.get('min_entity_length', 2)

    def get_max_entity_length(self) -> int:
        """获取最大实体长度"""
        config = self._configs.get('entity_extraction', {})
        return config.get('max_entity_length', 50)

    def get_max_extracted_entities(self) -> int:
        """获取最大提取实体数量"""
        config = self._configs.get('entity_extraction', {})
        return config.get('max_extracted_entities', 20)

    def is_alias_matching_enabled(self) -> bool:
        """是否启用别名匹配"""
        config = self._configs.get('entity_extraction', {})
        return config.get('enable_alias_matching', True)

    def get_alias_matching_strategy(self) -> str:
        """获取别名匹配策略"""
        config = self._configs.get('entity_extraction', {})
        return config.get('alias_matching_strategy', 'hybrid')

    def is_case_insensitive(self) -> bool:
        """是否忽略大小写"""
        config = self._configs.get('entity_extraction', {})
        return config.get('case_insensitive', True)

    def is_new_entity_extraction_enabled(self) -> bool:
        """是否启用新实体提取"""
        config = self._configs.get('entity_extraction', {})
        return config.get('enable_new_entity_extraction', True)

    # ========== 实体模式配置 ==========

    def get_entity_patterns(self) -> Dict[str, Any]:
        """获取实体类型模式"""
        patterns = self._configs.get('entity_patterns', {})
        return patterns.get('entity_patterns', {})

    def get_compiled_entity_patterns(self) -> Dict[str, List[re.Pattern]]:
        """获取编译后的实体模式"""
        return self._compiled_entity_patterns

    def get_type_priority_order(self) -> List[str]:
        """获取实体类型优先级排序"""
        config = self._configs.get('entity_patterns', {})
        priority_order = config.get('type_priority', [])
        if not priority_order:
            # 如果没有配置优先级，按重要性排序
            types = self.get_entity_types()
            priority_order = sorted(types.keys(), key=lambda t: types[t].get('priority', 999))
        return priority_order

    def get_max_entities_per_doc(self) -> int:
        """获取每个文档最大实体数量"""
        config = self._configs.get('entity_patterns', {})
        return config.get('max_entities', 100)

    # ========== 关系模式配置 ==========

    def get_relation_patterns_list(self) -> List[Dict]:
        """获取关系模式列表"""
        return self._configs.get('relation_patterns', {}).get('relation_patterns', [])

    def get_compiled_relation_patterns(self) -> Dict[str, str]:
        """获取编译后的关系模式 {pattern_str: relation_type}"""
        return self._compiled_relation_patterns

    def get_default_relation_weight(self) -> float:
        """获取默认关系权重"""
        config = self._configs.get('relation_patterns', {})
        return config.get('default_relation_weight', 1.0)

    def get_max_relation_weight(self) -> float:
        """获取最大关系权重"""
        config = self._configs.get('relation_patterns', {})
        return config.get('max_relation_weight', 5.0)

    # ========== 实体别名配置 ==========

    def get_entity_aliases(self) -> Dict[str, str]:
        """获取实体别名映射 {alias: full_name}"""
        config = self._configs.get('entity_aliases', {})
        aliases = config.get('entity_aliases', {})
        if aliases:
            return dict(sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True))
        return aliases

    def normalize_entity_name(self, name: str) -> str:
        """标准化实体名称（将别名转换为标准名）"""
        aliases = self.get_entity_aliases()
        if not aliases:
            return name

        case_insensitive = self.is_case_insensitive()
        compare_name = name.lower() if case_insensitive else name

        for alias, full_name in aliases.items():
            alias_compare = alias.lower() if case_insensitive else alias
            if alias_compare == compare_name:
                return full_name

        for alias, full_name in aliases.items():
            alias_compare = alias.lower() if case_insensitive else alias
            if alias_compare in compare_name:
                return full_name

        return name

    # ========== 查询替换配置 ==========

    def get_query_replacements(self) -> List[Tuple[str, str]]:
        """获取查询替换规则 [(alias, replacement), ...]"""
        config = self._configs.get('query_replacements', {})
        replacements = config.get('query_replacements', [])

        result = [(item.get('alias', ''), item.get('replacement', ''))
                  for item in replacements if item.get('alias') and item.get('replacement')]

        if config.get('sort_by_length', True):
            result.sort(key=lambda x: len(x[0]), reverse=True)

        return result

    def is_query_replacement_enabled(self) -> bool:
        """查询替换是否启用"""
        config = self._configs.get('query_replacements', {})
        return config.get('enable_query_replacement', True)

    def apply_query_replacements(self, query: str) -> str:
        """应用查询替换规则"""
        if not self.is_query_replacement_enabled():
            return query

        result = query
        for alias, replacement in self.get_query_replacements():
            result = result.replace(alias, replacement)
        return result

    # ========== 通用方法 ==========

    def get_raw_config(self, name: str) -> Dict:
        """获取原始配置"""
        return self._configs.get(name, {})

    def is_loaded(self) -> bool:
        """检查配置是否已加载"""
        return self._loaded and YAML_AVAILABLE

    def get_config_summary(self) -> Dict[str, int]:
        """获取配置摘要"""
        return {
            "entity_types": len(self.get_entity_types()),
            "relation_types": len(self.get_relation_types()),
            "entity_patterns": len(self.get_entity_patterns()),
            "relation_patterns": len(self.get_relation_patterns_list()),
            "entity_aliases": len(self.get_entity_aliases()),
            "query_replacements": len(self.get_query_replacements()),
            "synonyms": len(self.get_all_synonyms()),
            "entity_extraction_patterns": len(self.get_entity_extraction_patterns()),
        }


# 全局单例
_config: Optional[GraphRAGConfig] = None


def get_graphrag_config() -> GraphRAGConfig:
    """获取 GraphRAG 配置管理器实例"""
    global _config
    if _config is None:
        _config = GraphRAGConfig()
    return _config


def reload_config():
    """重新加载配置"""
    if _config:
        _config.reload()


__all__ = [
    'GraphRAGConfig',
    'get_graphrag_config',
    'reload_config',
]