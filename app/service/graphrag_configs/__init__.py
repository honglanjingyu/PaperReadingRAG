# app/service/graphrag_configs/__init__.py
"""
GraphRAG 配置加载模块
从 YAML 文件加载实体模式、关系模式、别名等配置
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML 未安装，请运行: pip install pyyaml")


class GraphRAGConfigLoader:
    """GraphRAG 配置加载器"""

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
        self._configs = {}
        self._loaded = False

        if not YAML_AVAILABLE:
            logger.error("PyYAML 未安装，无法加载配置")
            return

        self._load_all()

    def _load_all(self):
        """加载所有配置文件"""
        config_files = {
            'entity_patterns': self.config_dir / 'entity_patterns.yaml',
            'relation_patterns': self.config_dir / 'relation_patterns.yaml',
            'entity_aliases': self.config_dir / 'entity_aliases.yaml',
            'query_replacements': self.config_dir / 'query_replacements.yaml',
            'entity_extraction': self.config_dir / 'entity_extraction.yaml',  # 新增
        }

        for name, file_path in config_files.items():
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._configs[name] = yaml.safe_load(f) or {}
                    logger.info(f"加载配置: {name} -> {file_path}")
                except Exception as e:
                    logger.error(f"加载配置失败 {name}: {e}")
                    self._configs[name] = {}
            else:
                logger.warning(f"配置文件不存在: {file_path}")
                self._configs[name] = {}

        self._loaded = True

    def reload(self):
        """重新加载配置"""
        self._load_all()

    # ========== 实体提取相关（从 entity_extraction.yaml） ==========

    def get_entity_extraction_patterns(self) -> List[str]:
        """
        获取从问题中提取实体的正则表达式模式列表

        Returns:
            正则表达式模式列表
        """
        config = self._configs.get('entity_extraction', {})
        patterns = config.get('entity_extraction_patterns', [])

        return patterns

    def get_entity_length_limits(self) -> Tuple[int, int]:
        """
        获取实体长度限制（用于 entity_extractor）

        Returns:
            (min_entity_length, max_entity_length): 最小和最大实体长度
        """
        config = self._configs.get('entity_extraction', {})
        min_length = config.get('min_entity_length', 2)
        max_length = config.get('max_entity_length', 50)
        return min_length, max_length

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

    def is_new_entity_extraction_enabled(self) -> bool:
        """是否启用新实体提取"""
        config = self._configs.get('entity_extraction', {})
        return config.get('enable_new_entity_extraction', True)

    # ========== 实体模式相关（从 entity_patterns.yaml） ==========

    def get_entity_patterns(self) -> Dict[str, List[str]]:
        """获取实体类型模式"""
        patterns = self._configs.get('entity_patterns', {})
        return patterns.get('entity_patterns', {})

    def _compile_pattern(self, pattern_str: str) -> re.Pattern:
        """编译正则表达式，处理 YAML 中的转义"""
        try:
            pattern_str = pattern_str.replace('\\\\', '\\')
            pattern_str = pattern_str.encode('utf-8').decode('unicode_escape')
            return re.compile(pattern_str)
        except re.error as e:
            logger.error(f"编译正则失败: {pattern_str} -> {e}")
            return re.compile(r'.*')

    def get_compiled_entity_patterns(self) -> Dict[str, List[re.Pattern]]:
        """获取编译后的实体模式（正则对象）"""
        compiled = {}
        patterns = self.get_entity_patterns()

        for entity_type, pattern_list in patterns.items():
            compiled[entity_type] = []
            for pattern_str in pattern_list:
                compiled[entity_type].append(self._compile_pattern(pattern_str))

        return compiled

    def get_type_priority(self) -> List[str]:
        """获取实体类型优先级"""
        config = self._configs.get('entity_patterns', {})
        return config.get('type_priority', ['ORGANIZATION', 'TECHNOLOGY', 'PERSON'])

    def get_max_entities(self) -> int:
        """获取最大实体数量限制"""
        config = self._configs.get('entity_patterns', {})
        return config.get('max_entities', 100)

    # ========== 关系模式相关（从 relation_patterns.yaml） ==========

    def get_relation_patterns(self) -> List[Tuple[str, str]]:
        """获取关系模式列表 [(pattern_str, relation_type), ...]"""
        patterns = self._configs.get('relation_patterns', {})
        result = []

        for item in patterns.get('relation_patterns', []):
            pattern_str = item.get('pattern', '')
            relation_type = item.get('relation_type', 'RELATED_TO')
            if pattern_str:
                result.append((pattern_str, relation_type))

        return result

    def _compile_relation_pattern(self, pattern_str: str) -> re.Pattern:
        """编译关系正则表达式"""
        try:
            pattern_str = pattern_str.replace('\\\\', '\\')
            pattern_str = pattern_str.encode('utf-8').decode('unicode_escape')
            return re.compile(pattern_str)
        except re.error as e:
            logger.error(f"编译关系正则失败: {pattern_str} -> {e}")
            return re.compile(r'(.*?)')

    def get_compiled_relation_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """获取编译后的关系模式"""
        compiled = []
        patterns = self.get_relation_patterns()

        for pattern_str, relation_type in patterns:
            compiled.append((self._compile_relation_pattern(pattern_str), relation_type))

        return compiled

    def get_relation_direction(self) -> Dict[str, str]:
        """获取关系方向配置"""
        config = self._configs.get('relation_patterns', {})
        return config.get('relation_direction', {})

    def get_relation_weight_config(self) -> Dict[str, Any]:
        """获取关系权重配置"""
        config = self._configs.get('relation_patterns', {})
        return {
            'default_weight': config.get('default_relation_weight', 1.0),
            'max_weight': config.get('max_relation_weight', 5.0)
        }

    # ========== 实体别名相关（从 entity_aliases.yaml） ==========

    def get_entity_aliases(self) -> Dict[str, str]:
        """获取实体别名映射 {alias: full_name}"""
        config = self._configs.get('entity_aliases', {})
        aliases = config.get('entity_aliases', {})

        if aliases:
            return dict(sorted(aliases.items(), key=lambda x: len(x[0]), reverse=True))
        return aliases

    def get_alias_matching_config(self) -> Dict[str, Any]:
        """获取别名匹配配置"""
        config = self._configs.get('entity_aliases', {})
        return {
            'default_strategy': config.get('alias_matching', {}).get('default_strategy', 'exact'),
            'strategies': config.get('alias_matching', {}).get('strategies', {}),
            'case_insensitive': config.get('alias_matching', {}).get('case_insensitive', True)
        }

    def normalize_entity_name(self, name: str) -> str:
        """标准化实体名称（将别名转换为标准名）"""
        aliases = self.get_entity_aliases()
        config = self.get_alias_matching_config()
        case_insensitive = config.get('case_insensitive', True)

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

    def get_all_aliases(self) -> Dict[str, str]:
        """获取所有别名映射（用于图检索器）"""
        return self.get_entity_aliases()

    # ========== 查询替换相关（从 query_replacements.yaml） ==========

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

    # ========== 通用方法 ==========

    def get_raw_config(self, name: str) -> Dict:
        """获取原始配置"""
        return self._configs.get(name, {})


# 全局单例
_config_loader = None


def get_graphrag_config() -> GraphRAGConfigLoader:
    """获取 GraphRAG 配置加载器实例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = GraphRAGConfigLoader()
    return _config_loader


# 便捷函数
def get_entity_extraction_patterns() -> List[str]:
    """获取实体提取模式"""
    return get_graphrag_config().get_entity_extraction_patterns()


def get_min_entity_length() -> int:
    """获取最小实体长度"""
    return get_graphrag_config().get_min_entity_length()


def get_max_entity_length() -> int:
    """获取最大实体长度"""
    return get_graphrag_config().get_max_entity_length()


def get_max_extracted_entities() -> int:
    """获取最大提取实体数量"""
    return get_graphrag_config().get_max_extracted_entities()


__all__ = [
    'GraphRAGConfigLoader',
    'get_graphrag_config',
    'get_entity_extraction_patterns',
    'get_min_entity_length',
    'get_max_entity_length',
    'get_max_extracted_entities'
]