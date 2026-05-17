# app/service/graphrag_configs/__init__.py
"""
GraphRAG 配置加载模块
从 YAML 文件加载实体模式、关系模式、别名等配置
"""

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


# 从新模块导入配置类
from .loader import GraphRAGConfig, get_graphrag_config, reload_config

# ========== 便捷函数（供其他模块使用） ==========

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


# ========== 同义词相关函数 ==========

def get_all_synonyms() -> Dict[str, List[str]]:
    """获取所有同义词"""
    return get_graphrag_config().get_all_synonyms()


def get_synonyms(word: str) -> List[str]:
    """获取指定词的同义词"""
    return get_graphrag_config().get_synonyms(word)


def expand_with_synonyms(query: str) -> str:
    """使用同义词扩展查询"""
    return get_graphrag_config().expand_with_synonyms(query)


# ========== 实体别名相关 ==========

def get_entity_aliases() -> Dict[str, str]:
    """获取实体别名"""
    return get_graphrag_config().get_entity_aliases()


def normalize_entity_name(name: str) -> str:
    """标准化实体名称"""
    return get_graphrag_config().normalize_entity_name(name)


# ========== 类型定义相关 ==========

def get_entity_types() -> Dict[str, Dict]:
    """获取实体类型定义"""
    return get_graphrag_config().get_entity_types()


def get_relation_types() -> Dict[str, Dict]:
    """获取关系类型定义"""
    return get_graphrag_config().get_relation_types()


def get_entity_patterns() -> Dict[str, Any]:
    """获取实体模式配置"""
    return get_graphrag_config().get_entity_patterns()


def get_relation_patterns() -> List[Dict]:
    """获取关系模式配置"""
    return get_graphrag_config().get_relation_patterns_list()


def get_query_replacements() -> List[Tuple[str, str]]:
    """获取查询替换规则"""
    return get_graphrag_config().get_query_replacements()


def apply_query_replacements(query: str) -> str:
    """应用查询替换"""
    return get_graphrag_config().apply_query_replacements(query)


def get_config_summary() -> Dict[str, int]:
    """获取配置摘要"""
    return get_graphrag_config().get_config_summary()


__all__ = [
    # 核心类
    'GraphRAGConfig',
    'get_graphrag_config',
    'reload_config',
    # 实体提取
    'get_entity_extraction_patterns',
    'get_min_entity_length',
    'get_max_entity_length',
    'get_max_extracted_entities',
    # 同义词
    'get_all_synonyms',
    'get_synonyms',
    'expand_with_synonyms',
    # 实体别名
    'get_entity_aliases',
    'normalize_entity_name',
    # 类型定义
    'get_entity_types',
    'get_relation_types',
    'get_entity_patterns',
    'get_relation_patterns',
    # 查询替换
    'get_query_replacements',
    'apply_query_replacements',
    # 工具
    'get_config_summary',
]